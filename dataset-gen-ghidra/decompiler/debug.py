import os
import pickle

from ghidra.app.decompiler import DecompInterface

from collect import Collector
from ghidra_function import Function
from ghidra_types import TypeLib


class CollectDebug(Collector):
    """Class for collecting debug information"""

    def __init__(self):
        self.functions: Dict[int, Function] = dict()
        super().__init__()

    def write_functions(self) -> None:
        """Dumps the collected functions to the file specified by the environment
        variable `FUNCTIONS`.
        """
        with open(os.environ["FUNCTIONS"], "wb") as functions_fh:
            pickle.dump(self.functions, functions_fh)
            functions_fh.flush()

    def activate(self, ctx) -> int:
        """Collects types, user-defined variables, and their locations"""

        print("Collecting vars and types.")
        # `ea` is the start address of a single function
        decomp = DecompInterface()
        decomp.openProgram(currentProgram)

        # Ghidra separates Variables from their Data info, populate typelib first
        # for data in currentProgram.getListing().getDefinedData(True):
        #     self.type_lib.add_ghidra_type(data)

        for f in currentProgram.getListing().getFunctions(True):
            # Decompile
            decomp_results = decomp.decompileFunction(f, 30, None)

            if not decomp_results.decompileCompleted():
                continue

            if decomp_results.getErrorMessage() != "":
                continue

            # Function info
            name: str = f.getName()
            self.type_lib.add_ghidra_type(f.getReturnType())
            return_type = TypeLib.parse_ghidra_type(f.getReturnType())

            arguments = self.collect_variables(
                f.getStackFrame().getFrameSize(), f.getParameters()
            )
            local_vars = self.collect_variables(
                f.getStackFrame().getFrameSize(),
                # [v for v in f.getLocalVariables()],
                f.getLocalVariables()
            )
            self.functions[f.getEntryPoint().toString()] = Function(
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
            )

        self.write_type_lib()
        self.write_functions()
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

debug = CollectDebug()
debug.activate(None)
# ida.qexit(0)
