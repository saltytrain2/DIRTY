import gzip
import jsonlines
import os
import pickle

from typing import Dict, List

from ghidra.app.decompiler import DecompInterface

from ghidra_ast import AST
from collect import Collector
from ghidra_function import CollectedFunction, Function
from ghidra_types import TypeLib


class CollectDecompiler(Collector):
    """Class for collecting decompiler-specific information"""

    def __init__(self):
        print("Initializing collect decompiler")
        super().__init__()
        print("Loading functions")
        # Load the functions collected by CollectDebug
        with open(os.environ["FUNCTIONS"], "rb") as functions_fh:
            self.debug_functions: Dict[int, Function] = pickle.load(functions_fh)
        print("Done")
        self.functions: List[CollectedFunction] = list()
        self.output_file_name = os.path.join(
            os.environ['OUTPUT_DIR'],
            "bins",
            os.environ['PREFIX'] + ".jsonl.gz",
        )

    def write_info(self) -> None:
        with gzip.open(self.output_file_name, 'wt') as output_file:
            with jsonlines.Writer(output_file, compact=True) as writer:
                for cf in self.functions:
                    writer.write(cf.to_json())

    def activate(self, ctx) -> int:
        """Collects types, user-defined variables, their locations in addition to the
        AST and raw code.
        """
        print("Collecting vars and types.")

        decomp = DecompInterface()
        decomp.toggleSyntaxTree(False)
        decomp.openProgram(currentProgram)

        for f in currentProgram.getListing().getFunctions(True):
            # Decompile
            decomp_results = decomp.decompileFunction(f, 30, None)
            f = decomp_results.getFunction()

            if not decomp_results.decompileCompleted():
                continue

            if decomp_results.getErrorMessage() != "":
                continue
            
            high_func = decomp_results.getHighFunction()
            lsm = high_func.getLocalSymbolMap()
            symbols = [v for v in lsm.getSymbols()]
            func_return = high_func.getFunctionPrototype().getReturnType()

            name:str = f.getName()
            self.type_lib.add_ghidra_type(func_return)
            return_type = TypeLib.parse_ghidra_type(func_return)

            arguments = self.collect_variables(
                f.getStackFrame().getFrameSize(), [v for v in symbols if v.isParameter()],
            )
            local_vars = self.collect_variables(
                f.getStackFrame().getFrameSize(),
                [v for v in symbols if not v.isParameter()],
            )

            #raw_code = decomp_results.getDecompiledFunction().getC()
            raw_code = decomp_results.getCCodeMarkup().toString()

            decompiler = Function(
                ast=None,
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
                raw_code=raw_code,
            )
            self.functions.append(
                CollectedFunction(
                    ea=f.getEntryPoint().toString(),
                    debug=self.debug_functions[f.getEntryPoint().toString()],
                    decompiler=decompiler,
                )
            )
        self.write_info()
        print("Done with dump_trees")
        return 1


# ida.auto_wait()
# if not ida.init_hexrays_plugin():
#     ida.load_plugin("hexrays")
#     ida.load_plugin("hexx64")
#     if not ida.init_hexrays_plugin():
#         print("Unable to load Hex-rays")
#         ida.qexit(1)
#     else:
#         print(f"Hex-rays version {ida.get_hexrays_version()}")

decompiler = CollectDecompiler()
decompiler.activate(None)
print("Done with activate")
# ida.qexit(0)