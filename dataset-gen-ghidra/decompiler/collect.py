import gzip
import os

from collections import defaultdict
from typing import DefaultDict, Iterable, Optional, Set

from ghidra_types import TypeInfo, TypeLib, TypeLibCodec
from ghidra_variable import Location, Stack, Register, Variable

import ghidra.program.model.symbol

class Collector:
    """Generic class to collect information from a binary"""

    def __init__(self):
        # Load the type library
        self.type_lib_file_name = os.path.join(
            os.environ["OUTPUT_DIR"],
            "types",
            os.environ["PREFIX"] + ".json.gz",
        )
        try:
            with gzip.open(self.type_lib_file_name, "rt") as type_lib_file:
                self.type_lib = TypeLibCodec.decode(type_lib_file.read())
        except Exception as e:
            print(e)
            print("Could not find type library, creating a new one")
            self.type_lib = TypeLib()

    def write_type_lib(self) -> None:
        """Dumps the type library to the file specified by the environment variable
        `TYPE_LIB`.
        """
        with gzip.open(self.type_lib_file_name, "wt") as type_lib_file:
            encoded = TypeLibCodec.encode(self.type_lib)
            type_lib_file.write(encoded)
            type_lib_file.flush()

    def collect_variables(
        self, frsize: int, variables:Iterable,
    ) -> DefaultDict[Location, Set[Variable]]:
        """Collects Variables from a list of HighSymbols and adds their types to the type
        library."""
        collected_vars: DefaultDict[Location, Set[Variable]] = defaultdict(set)
        for v in variables:
            if v.getName() == "":
                continue

            # Add all types to the typelib
            self.type_lib.add_ghidra_type(v.getDataType())
            typ: TypeInfo = TypeLib.parse_ghidra_type(v.getDataType())

            loc: Optional[Location] = None
            storage = v.getStorage()

            has_user_info = False
            low_symbol = v.getSymbol()
            if low_symbol is not None and low_symbol.getSource() in [ghidra.program.model.symbol.SourceType.IMPORTED, ghidra.program.model.symbol.SourceType.USER_DEFINED]:
                has_user_info = True

            if storage.isStackStorage():
                loc = Stack(storage.getStackOffset())
            if storage.isRegisterStorage():
                loc = Register(storage.getRegister().getName())
            if loc is not None:
                collected_vars[loc].add(
                    Variable(typ=typ, name=v.getName(), user=has_user_info)
                )
        return collected_vars

    def activate(self, ctx) -> int:
        """Runs the collector"""
        raise NotImplementedError