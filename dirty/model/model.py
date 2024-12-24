import json
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional.classification # for multiclass_accuracy
from utils.vocab import Vocab
from utils.ghidra_types import TypeInfo, TypeLibCodec

from model.encoder import Encoder
from model.decoder import Decoder

# Wow, macro is the default.  That is crazy.
def accuracy(preds, targets, average="micro", **kwargs):
    if "num_classes" not in kwargs and average == "micro":
        kwargs["num_classes"] = len(targets) # doesn't matter for micro
    return torchmetrics.functional.classification.multiclass_accuracy(preds, targets, average=average, **kwargs)

class RenamingDecodeModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder.build({**config["decoder"], "rename": True})
        self.soft_mem_mask = config["decoder"]["mem_mask"] == "soft"
        if self.soft_mem_mask:
            self.mem_encoder = Encoder.build(config["mem_encoder"])
            self.mem_decoder = Decoder.build(config["mem_decoder"])
            self.decoder.mem_encoder = self.mem_encoder
            self.decoder.mem_decoder = self.mem_decoder
        self.beam_size = config["test"]["beam_size"]

    def training_step(self, input_dict, context_encoding, target_dict):
        variable_name_logits = self.decoder(context_encoding, target_dict)
        if self.soft_mem_mask:
            variable_name_logits = variable_name_logits[target_dict["target_type_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                variable_name_logits + mem_logits,
                target_dict["target_name_id"][target_dict["target_type_mask"]],
                reduction="none",
            )
        else:
            loss = F.cross_entropy(
                # cross_entropy requires num_classes at the second dimension
                variable_name_logits.transpose(1, 2),
                target_dict["target_name_id"],
                reduction="none",
            )
            loss = loss[target_dict["target_type_mask"]]
        return loss.mean()

    def shared_eval_step(self, context_encoding, input_dict, target_dict, test=False):
        variable_name_logits = self.decoder(context_encoding, target_dict)
        if self.soft_mem_mask:
            variable_name_logits = variable_name_logits[input_dict["src_type_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                variable_name_logits + mem_logits,
                target_dict["target_name_id"][input_dict["src_type_mask"]],
                reduction="none",
            )
        else:
            loss = F.cross_entropy(
                variable_name_logits.transpose(1, 2),
                target_dict["target_name_id"],
                reduction="none",
            )
            loss = loss[input_dict["src_type_mask"]]
        targets = target_dict["target_name_id"][input_dict["src_type_mask"]]
        preds = self.decoder.predict(
            context_encoding, input_dict, None, self.beam_size if test else 0
        )

        return dict(
            rename_loss=loss.detach().cpu(),
            rename_preds=preds.detach().cpu(),
            rename_targets=targets.detach().cpu(),
        )


class RetypingDecodeModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder.build({**config["decoder"]})
        self.subtype = config["decoder"]["type"] in ["XfmrSubtypeDecoder"]
        self.soft_mem_mask = config["decoder"]["mem_mask"] == "soft"
        if self.soft_mem_mask:
            self.mem_encoder = Encoder.build(config["mem_encoder"])
            self.mem_decoder = Decoder.build(config["mem_decoder"])
            self.decoder.mem_encoder = self.mem_encoder
            self.decoder.mem_decoder = self.mem_decoder
        self.beam_size = config["test"]["beam_size"]

    def training_step(self, input_dict, context_encoding, target_dict):
        variable_type_logits = self.decoder(context_encoding, target_dict)
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[target_dict["target_type_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][target_dict["target_type_mask"]],
                reduction="none",
            )
        else:
            loss = F.cross_entropy(
                variable_type_logits.transpose(1, 2),
                target_dict["target_subtype_id"]
                if self.subtype
                else target_dict["target_type_id"],
                reduction="none",
            )
            loss = loss[
                target_dict["target_submask"]
                if self.subtype
                else target_dict["target_type_mask"]
            ]

        return loss.mean()

    def shared_eval_step(self, context_encoding, input_dict, target_dict, test=False):
        variable_type_logits = self.decoder(context_encoding, target_dict)
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[input_dict["src_type_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            loss = F.cross_entropy(
                # cross_entropy requires num_classes at the second dimension
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][input_dict["src_type_mask"]],
                reduction="none",
            )
        else:
            loss = F.cross_entropy(
                # cross_entropy requires num_classes at the second dimension
                variable_type_logits.transpose(1, 2),
                target_dict["target_subtype_id"]
                if self.subtype
                else target_dict["target_type_id"],
                reduction="none",
            )
            loss = loss[
                target_dict["target_submask"]
                if self.subtype
                else target_dict["target_type_mask"]
            ]
        targets = target_dict["target_type_id"][input_dict["src_type_mask"]]
        preds = self.decoder.predict(
            context_encoding, input_dict, None, self.beam_size if test else 0
        )

        return dict(
            retype_loss=loss.detach().cpu(),
            retype_preds=preds.detach().cpu(),
            retype_targets=targets.detach().cpu(),
        )


class InterleaveDecodeModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder.build({**config["decoder"]})
        self.soft_mem_mask = config["decoder"]["mem_mask"] == "soft"
        self.beam_size = config["test"]["beam_size"]
        if self.soft_mem_mask:
            self.mem_encoder = Encoder.build(config["mem_encoder"])
            self.mem_decoder = Decoder.build(config["mem_decoder"])
            self.decoder.mem_encoder = self.mem_encoder
            self.decoder.mem_decoder = self.mem_decoder

    def training_step(self, input_dict, context_encoding, target_dict):
        variable_type_logits, variable_name_logits = self.decoder(
            context_encoding, target_dict
        )
        # Retype
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[target_dict["target_type_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            retype_loss = F.cross_entropy(
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][target_dict["target_type_mask"]],
                reduction="none",
            )
        else:
            retype_loss = F.cross_entropy(
                variable_type_logits.transpose(1, 2),
                target_dict["target_type_id"],
                reduction="none",
            )
            retype_loss = retype_loss[target_dict["target_type_mask"]]
        retype_loss = retype_loss.mean()

        rename_loss = F.cross_entropy(
            # cross_entropy requires num_classes at the second dimension
            variable_name_logits.transpose(1, 2),
            target_dict["target_name_id"],
            reduction="none",
        )
        rename_loss = rename_loss[target_dict["target_type_mask"]].mean()

        return retype_loss, rename_loss

    def forward(self, context_encoding, input_dict, **kwargs):
        return self.decoder.predict(context_encoding, input_dict, self.beam_size, **kwargs)

    def get_unmasked_logits(self, context_encoding, input_dict, target_dict):
        variable_type_logits, _ = self.decoder(context_encoding, target_dict)
        variable_type_logits = variable_type_logits[target_dict["target_type_mask"]]
        # mem_encoding = self.mem_encoder(input_dict)
        # return self.mem_decoder(mem_encoding, target_dict) + variable_type_logits
        return variable_type_logits.argmax(dim=1)

    def shared_eval_step(self, context_encoding, input_dict, target_dict, test=False):
        variable_type_logits, variable_name_logits = self.decoder(
            context_encoding, target_dict
        )
        if self.soft_mem_mask:
            variable_type_logits = variable_type_logits[input_dict["src_type_mask"]]
            mem_encoding = self.mem_encoder(input_dict)
            mem_type_logits = self.mem_decoder(mem_encoding, target_dict)
            retype_loss = F.cross_entropy(
                variable_type_logits + mem_type_logits,
                target_dict["target_type_id"][input_dict["src_type_mask"]],
                reduction="none",
            )
        else:
            retype_loss = F.cross_entropy(
                variable_type_logits.transpose(1, 2),
                target_dict["target_type_id"],
                reduction="none",
            )
            retype_loss = retype_loss[target_dict["target_type_mask"]]

        rename_loss = F.cross_entropy(
            variable_name_logits.transpose(1, 2),
            target_dict["target_name_id"],
            reduction="none",
        )
        rename_loss = rename_loss[input_dict["src_type_mask"]]
        ret = self.decoder.predict(
            context_encoding, input_dict, self.beam_size if test else 0
        )
        retype_preds, rename_preds = ret[0], ret[1]

        return dict(
            retype_loss=retype_loss.detach().cpu(),
            retype_targets=target_dict["target_type_id"][input_dict["src_type_mask"]]
            .detach()
            .cpu(),
            retype_preds=retype_preds.detach().cpu(),
            rename_loss=rename_loss.detach().cpu(),
            rename_targets=target_dict["target_name_id"][input_dict["src_type_mask"]]
            .detach()
            .cpu(),
            rename_preds=rename_preds.detach().cpu(),
        )


class TypeReconstructionModel(pl.LightningModule):
    def __init__(self, config, config_load=None):
        super().__init__()
        if config_load is not None:
            config = config_load
        # Lame, we need to save our outputs now!
        # https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        self.eval_outputs = []
        self.encoder = Encoder.build(config["encoder"])
        self.retype = config["data"].get("retype", False)
        self.rename = config["data"].get("rename", False)
        self.interleave = config["data"].get("interleave", False)
        if self.interleave:
            self.interleave_module = InterleaveDecodeModule(config)
        else:
            if self.retype:
                self.retyping_module = RetypingDecodeModule(config)
            if self.rename:
                self.renaming_module = RenamingDecodeModule(config)
        self.config = config
        self.vocab = Vocab.load(config["data"]["vocab_file"])
        self._preprocess()
        self.soft_mem_mask = config["decoder"]["mem_mask"] == "soft"

    def _preprocess(self):
        self.vocab.types.struct_set = set()
        for idx, type_str in self.vocab.types.id2word.items():
            if type_str.startswith("struct"):
                self.vocab.types.struct_set.add(idx)
        with open(self.config["data"]["typelib_file"]) as type_f:
            typelib = TypeLibCodec.decode(type_f.read())
            self.typstr_to_piece = {}
            for size in typelib:
                for _, tp in typelib[size]:
                    self.typstr_to_piece[str(tp)] = tp.tokenize()[:-1]
        self.typstr_to_piece["<unk>"] = ["<unk>"]

    def training_step(
        self,
        batch: Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, torch.Tensor]],
        batch_idx,
    ):
        input_dict, target_dict = batch
        total_loss = 0
        context_encoding = self.encoder(input_dict)
        if self.interleave:
            retype_loss, rename_loss = self.interleave_module.training_step(
                input_dict, context_encoding, target_dict
            )
            self.log("train_retype_loss", retype_loss)
            self.log("train_rename_loss", rename_loss)
            total_loss = retype_loss + rename_loss
        else:
            if self.retype:
                loss = self.retyping_module.training_step(
                    input_dict, context_encoding, target_dict
                )
                self.log("train_retype_loss", loss)
                total_loss += loss
            if self.rename:
                loss = self.renaming_module.training_step(
                    input_dict, context_encoding, target_dict
                )
                self.log("train_rename_loss", loss)
                total_loss += loss
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, test=True)

    def get_unmasked_logits(self, batch):
        input_dict, target_dict = batch
        context_encoding = self.encoder(input_dict)
        return self.interleave_module.get_unmasked_logits(context_encoding, input_dict, target_dict)

    def _shared_eval_step(
        self,
        batch: Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, torch.Tensor]],
        batch_idx,
        test=False,
    ):
        input_dict, target_dict = batch
        context_encoding = self.encoder(input_dict)
        ret_dict = {}
        if self.interleave:
            ret_dict = self.interleave_module.shared_eval_step(
                context_encoding, input_dict, target_dict, test
            )
        else:
            if self.retype:
                ret = self.retyping_module.shared_eval_step(
                    context_encoding, input_dict, target_dict, test
                )
                ret_dict = {**ret, **ret_dict}
            if self.rename:
                ret = self.renaming_module.shared_eval_step(
                    context_encoding, input_dict, target_dict, test
                )
                ret_dict = {**ret, **ret_dict}

        d = dict(
            **ret_dict,
            # this is the number of variables per example, which is same in src
            # and tgt.
            targets_nums=input_dict["src_type_mask"].sum(dim=1),
            test_meta=target_dict["test_meta"],
            index=input_dict["index"],
            tgt_var_names=target_dict["tgt_var_names"],
        )

        self.eval_outputs.append(d)

        return d
    
    def forward(self, batch, return_non_best=False):
        input_dict = batch
        context_encoding = self.encoder(input_dict)
        if self.interleave:
            ret = self.interleave_module(context_encoding, input_dict, return_non_best=return_non_best)
        else:
            if self.retype:
                ret = self.retyping_module(context_encoding, input_dict)
            elif self.rename:
                ret = self.renaming_module(context_encoding, input_dict)
            else:
                assert False

        if return_non_best:
            retype_preds, rename_preds, all_retype_preds, all_rename_preds = ret
        else:
            retype_preds, rename_preds = ret

        retype_preds_name = [self.vocab.types.id2word[x.item()] for x in retype_preds]

        rename_preds_name = [self.vocab.names.id2word[x.item()] for x in rename_preds]

        ret = {
            "retype_preds": retype_preds_name,
            "rename_preds": rename_preds_name,
        }

        if return_non_best:

            all_retype_preds_names = [
                [self.vocab.types.id2word[prediction.item()] for prediction in predictions] 
                for predictions in all_retype_preds
            ]

            all_rename_preds_names = [
                [self.vocab.names.id2word[prediction.item()] for prediction in predictions]
                for predictions in all_rename_preds
            ]

            ret.update({
                "all_retype_preds": all_retype_preds_names,
                "all_rename_preds": all_rename_preds_names,
            })

        return ret

    def on_validation_epoch_end(self):
        outputs = self.eval_outputs
        self._shared_epoch_end(outputs, "val")
        self.eval_outputs.clear()

    def on_test_epoch_end(self):
        outputs = self.eval_outputs
        final_ret = self._shared_epoch_end(outputs, "test")
        if "pred_file" in self.config["test"]:
            results = {}
            for (binary, func_name, decom_var_name), retype_pred, rename_pred in zip(
                final_ret["indexes"],
                final_ret["retype_preds"].tolist()
                if "retype_preds" in final_ret
                else [None] * len(final_ret["indexes"]),
                final_ret["rename_preds"].tolist()
                if "rename_preds" in final_ret
                else [None] * len(final_ret["indexes"]),
            ):
                results.setdefault(binary, {}).setdefault(func_name, {})[
                    decom_var_name[2:-2]
                ] = self.vocab.types.id2word.get(
                    retype_pred, ""
                ), self.vocab.names.id2word.get(
                    rename_pred, ""
                )
            pred_file = self.config["test"]["pred_file"]
            json.dump(results, open(pred_file, "w"))
        self.eval_outputs.clear()

    def _shared_epoch_end(self, outputs, prefix):
        final_ret = {}
        if self.retype:
            ret = self._shared_epoch_end_task(outputs, prefix, "retype")
            final_ret = {**final_ret, **ret}
            acc = final_ret["retype_acc"]
            loss = final_ret["retype_loss"]
        if self.rename:
            ret = self._shared_epoch_end_task(outputs, prefix, "rename")
            final_ret = {**final_ret, **ret}
            acc = final_ret["rename_acc"]
            loss = final_ret["rename_loss"]
        if self.retype and self.rename:
            acc = final_ret["retype_acc"] + final_ret["rename_acc"]
            loss = final_ret["retype_loss"] + final_ret["rename_loss"]
            self.log(f"{prefix}_acc", acc, sync_dist=True)
            self.log(f"{prefix}_loss", loss, sync_dist=True)
            # Evaluate rename accuracy on correctedly retyped samples
            retype_preds = torch.cat([x[f"retype_preds"] for x in outputs])
            retype_targets = torch.cat([x[f"retype_targets"] for x in outputs])
            rename_preds = torch.cat([x[f"rename_preds"] for x in outputs])
            rename_targets = torch.cat([x[f"rename_targets"] for x in outputs])
            if (retype_preds == retype_targets).sum() > 0:
                self.log(
                    f"{prefix}_rename_on_correct_retype_acc",
                    accuracy(
                        rename_preds[retype_preds == retype_targets],
                        rename_targets[retype_preds == retype_targets]
                    ),
                    sync_dist=True
                )

        return final_ret

    def _shared_epoch_end_task(self, outputs, prefix, task):
        indexes = sum([x["index"] for x in outputs], [])
        tgt_var_names = sum([x["tgt_var_names"] for x in outputs], [])
        preds = torch.cat([x[f"{task}_preds"] for x in outputs])
        targets = torch.cat([x[f"{task}_targets"] for x in outputs])
        loss = torch.cat([x[f"{task}_loss"] for x in outputs]).mean()
        self.log(f"{prefix}_{task}_loss", loss, sync_dist=True)
        acc = accuracy(preds, targets)
        self.log(f"{prefix}_{task}_acc", acc, sync_dist=True)
        self.log(
            f"{prefix}_{task}_acc_macro",
            accuracy(
                preds,
                targets,
                num_classes=len(self.vocab.types if task == "retype" else self.vocab.names),
                average="macro",
            ),
            sync_dist=True
        )
        # func acc
        num_correct, num_funcs, pos = 0, 0, 0
        body_in_train_mask = []
        name_in_train_mask = []
        for target_num, test_metas in map(
            lambda x: (x["targets_nums"], x["test_meta"]), outputs
        ):
            for num, test_meta in zip(target_num.tolist(), test_metas):
                num_correct += all(preds[pos : pos + num] == targets[pos : pos + num])
                pos += num
                body_in_train_mask += [test_meta["function_body_in_train"]] * num
                name_in_train_mask += [test_meta["function_name_in_train"]] * num
            num_funcs += len(target_num)
        body_in_train_mask = torch.tensor(body_in_train_mask)
        name_in_train_mask = torch.tensor(name_in_train_mask)
        if body_in_train_mask.dim() > 1:
            # HACK for data parallel
            body_in_train_mask = body_in_train_mask[:, 0]
            name_in_train_mask = name_in_train_mask[:, 0]
        if body_in_train_mask.sum() > 0:
            self.log(
                f"{prefix}_{task}_body_in_train_acc",
                accuracy(preds[body_in_train_mask], targets[body_in_train_mask]),
                sync_dist=True
            )
        if (~body_in_train_mask).sum() > 0:
            self.log(
                f"{prefix}_{task}_body_not_in_train_acc",
                accuracy(preds[~body_in_train_mask], targets[~body_in_train_mask]),
                sync_dist=True
            )
        assert pos == sum(x["targets_nums"].sum() for x in outputs), (
            pos,
            sum(x["targets_nums"].sum() for x in outputs),
        )
        self.log(f"{prefix}_{task}_func_acc", num_correct / num_funcs, sync_dist=True)

        struc_mask = torch.zeros(len(targets), dtype=torch.bool)
        for idx, target in enumerate(targets):
            if target.item() in self.vocab.types.struct_set:
                struc_mask[idx] = 1
        task_str = "" if task == "retype" else f"_{task}"
        if struc_mask.sum() > 0:
            self.log(
                f"{prefix}{task_str}_struc_acc",
                accuracy(preds[struc_mask], targets[struc_mask]),
                sync_dist=True
            )
            # adjust for the number of classes
            self.log(
                f"{prefix}{task_str}_struc_acc_macro",
                accuracy(
                    preds[struc_mask],
                    targets[struc_mask],
                    num_classes=len(self.vocab.types if task == "retype" else self.vocab.names),
                    average="macro",
                )
                * len(self.vocab.types)
                / len(self.vocab.types.struct_set),
                sync_dist=True
            )
        if (struc_mask & body_in_train_mask).sum() > 0:
            self.log(
                f"{prefix}{task_str}_body_in_train_struc_acc",
                accuracy(
                    preds[struc_mask & body_in_train_mask],
                    targets[struc_mask & body_in_train_mask]
                ),
                sync_dist=True
            )
        if (~body_in_train_mask & struc_mask).sum() > 0:
            self.log(
                f"{prefix}{task_str}_body_not_in_train_struc_acc",
                accuracy(
                    preds[~body_in_train_mask & struc_mask],
                    targets[~body_in_train_mask & struc_mask]
                ),
                sync_dist=True
            )
        return {
            "indexes": indexes,
            "tgt_var_names": tgt_var_names,
            f"{task}_preds": preds,
            f"{task}_targets": preds,
            "body_in_train_mask": body_in_train_mask,
            f"{task}_acc": acc,
            f"{task}_loss": loss
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config
        ["train"]["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1 # epoch
            }
        }