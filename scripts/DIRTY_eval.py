# Evaluation helper script for DIRTY Ghidra
from functools import lru_cache
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.pcode import HighFunctionDBUtil
from ghidra.program.model.symbol import SourceType
from ghidra.program.model.data import PointerDataType, ArrayDataType, StructureDataType, UnionDataType, TypedefDataType
from ghidra.app.services import DataTypeManagerService
from ghidra.app.plugin.core.analysis import AutoAnalysisManager
import json
import random

import ghidra_types

debug = False

def abort(s):
    raise Exception(s)

TYPELIB_PATH = "/home/ed/Downloads/typelib.json"

codec = ghidra_types.TypeLibCodec()
typelib = codec.decode(open(TYPELIB_PATH, "r").read())

name_to_type = {}

def all_typenames():
    for _size, typelist in typelib.items():
        for _freq, typeentry in typelist:
            yield str(typeentry)

def find_types_by_name(name):
    for size, typelist in typelib.items():
        for _freq, typeentry in typelist:
            typename = str(typeentry)
            #if size == 160000: # or "double" in typename:
            #if debug and name in typename:
            #    print(f"Hmm: {name} ==? {typename}")
            if typename == name:
                yield typeentry
    return

def find_type_by_name(name):
    try:
        return next(find_types_by_name(name))
    except StopIteration:
        print(f"Unable to find type {name} in typelib. Hopefully it is a built-in!")
        return ghidra_types.TypeInfo(name=name, size=0)

def find_type_in_ghidra_typemanager(name, dtm):
    if dtm is None:
        dtm = currentProgram().getDataTypeManager()

    al = dtm.getAllDataTypes()
    output_list = [dt for dt in al if dt.getName() == name]

    if len(output_list) > 0:
        return output_list[0]
    else:
        return None

def find_type_in_any_ghidra_typemanager(name):
    #dtms = [AutoAnalysisManager.getAnalysisManager(currentProgram()).getDataTypeManagerService()]
    dtms = [currentProgram().getDataTypeManager()]
    #dtms = state().getTool().getService(DataTypeManagerService).getDataTypeManagers()
    for dtm in dtms:
        output = find_type_in_ghidra_typemanager(name, dtm)
        if output is not None:
            return output
    return None

@lru_cache(maxsize=20000)
def build_ghidra_type(typelib_type):
    #print(f"build_ghidra_type {typelib_type} {typelib_type.__dict__} {type(typelib_type)}")

    # First check our cache.  This is important for self-referential types
    # (e.g., linked lists).
    if str(typelib_type) in name_to_type:
        return name_to_type[str(typelib_type)]

    out = find_type_in_any_ghidra_typemanager(str(typelib_type))
    if out is not None:
        return out

    if type(typelib_type) == ghidra_types.TypeInfo:

        print(f"WARNING: {typelib_type.name} is a TypeInfo type: {typelib_type.debug}")
        print(f"WARNING: Unable to find type {typelib_type.name} in Ghidra.")

        match typelib_type.size:
            case 1:
                return find_type_in_any_ghidra_typemanager("byte")
            case 4:
                return find_type_in_any_ghidra_typemanager("uint32_t")
            case 8:
                return find_type_in_any_ghidra_typemanager("uint64_t")
            case _:
                abort(f"Unknown type with unusual size: {typelib_type.size} {typelib_type.name}")

    elif type(typelib_type) == ghidra_types.Array:
        element_type = build_ghidra_type(find_type_by_name(typelib_type.element_type))
        return ArrayDataType(element_type, typelib_type.nelements, typelib_type.element_size)
    elif type(typelib_type) == ghidra_types.Pointer:
        target_type = build_ghidra_type(find_type_by_name(typelib_type.target_type_name))
        return PointerDataType(target_type)
        # Make type.
    elif type(typelib_type) == ghidra_types.Struct or type(typelib_type) == ghidra_types.Union:
        new_struct = StructureDataType(typelib_type.name, typelib_type.size) if type(typelib_type) == ghidra_types.Struct else UnionDataType(typelib_type.name)
        # We need to immediately make this available in case we have a self-referential type.
        name_to_type[str(typelib_type)] = new_struct
        offset = 0
        for member in (typelib_type.layout if type(typelib_type) == ghidra_types.Struct else typelib_type.members):
            if type(member) == ghidra_types.UDT.Padding:
                # Don't do anything?
                pass
                # new_struct.insertAtOffset(offset, VoidDataType(), member.size)
            elif type(member) == ghidra_types.UDT.Field:
                member_type = build_ghidra_type(find_type_by_name(member.type_name))
                if type(typelib_type) == ghidra_types.Struct:
                    new_struct.insertAtOffset(offset, member_type, member.size, member.name, "")
                elif type(typelib_type) == ghidra_types.Union:
                    new_struct.add(member_type, member.size, member.name, "")
                else:
                    abort("Unknown member type: " + str(type(typelib_type)))
            else:
                abort("Unknown member type: " + str(type(member)))

            offset = offset + member.size
                
            #field_type = build_ghidra_type(find_type_by_name(field.type))
            #new_struct.add(field_type, field.name, field.comment)
        return new_struct
    elif type(typelib_type) == ghidra_types.TypeDef:
        other_type = build_ghidra_type(find_type_by_name(typelib_type.other_type_name))
        return TypedefDataType(typelib_type.name, other_type)
    else:
        abort(f"Unknown type: {type(typelib_type)} {typelib_type}")

def test_types():
    total = 0
    succ = 0

    l = list(all_typenames())
    random.shuffle(l)

    #l = ["longlong[20][10]"]
    #l = ["xen_string_string_map"]
    #debug = True

    for typename in l:
        print(f"Trying to build type {typename}")
        total += 1

        if monitor().isCancelled():
            break

        monitor().setMessage(f"Building type {typename}")

        try:
            ti = find_type_by_name(typename)
        except Exception as e:
            print(f"Failed to find type {typename} exception: {e}")
            continue

        try:
            gtype = build_ghidra_type(ti)
            assert gtype is not None, "build_ghidra_type returned None."
            print(f"Successfully built type {typename} in Ghidra: {gtype}")
        except Exception as e:
            print(f"Failed to build ghidra type {typename} exception: {e}")
            #break
            continue

        succ += 1

        print(f"Successfully built {succ}/{total} {float(succ)/total} types.")

    exit(0)

print ([x for x in getScriptArgs()])

#exeName = currentProgram().getName()

jsonFile = getScriptArgs()[0]
exeName = getScriptArgs()[1]
targetFunction = getScriptArgs()[2]
targetFunctionAddr = int(targetFunction, 16)
outputJsonFile = getScriptArgs()[3]

#print("Using JSON file: " + jsonFile)
#print("Target function: " + targetFunctionAddr)

#jsonFile = askFile("Select JSON file", "Open")
print(f"Parsing JSON for {exeName}")
jsonObj = json.load(open(jsonFile))

if exeName in jsonObj:
    jsonObj = jsonObj[exeName]
elif len(jsonObj) == 1:
    jsonObj = jsonObj[list(jsonObj.keys())[0]]
else:
    abort(f"Unable to find the executable {exeName} in the JSON file.")

#current_location = currentLocation()

#for f in currentProgram().getFunctionManager().getFunctions(True):
#    print(f.getName())

func = next(f for f in currentProgram().getFunctionManager().getFunctions(True) if f.getEntryPoint().getOffset() == targetFunctionAddr)

if func is None:
    abort("Unable to find function " + targetFunction)

print("Found target function: " + func.getName())

assert func is not None

funcName = f"FUN_%08x" % targetFunctionAddr
if funcName in jsonObj:
    jsonObj = jsonObj[funcName]
else:
    abort(f"Unable to find function %s in the JSON file." % funcName)

 # Set up the decompiler
decompiler = DecompInterface()
decompiler.openProgram(func.getProgram())

# Decompile the current function
def decompile(func):
    print("Decompiling function " + func.getName() + "...")
    results = decompiler.decompileFunction(func, 0, ConsoleTaskMonitor())
    if not results.decompileCompleted():
        abort("Decompilation failed.")

    # Get the high-level representation of the function
    high_function = results.getHighFunction()
    if not high_function:
        abort("Failed to get high-level function representation.")

    return results, high_function

results, high_function = decompile(func)

output = {}

output['decompilation'] = results.getDecompiledFunction().getC()
output['vars'] = [v.getName() for v in high_function.getLocalSymbolMap().getSymbols()]
output['params'] = [v.getName() for v in high_function.getLocalSymbolMap().getSymbols() if v.isParameter()]

# Rename variables (if no symbols)
for var in high_function.getLocalSymbolMap().getSymbols():

    original_name = var.getName()

    if original_name in jsonObj:
        new_type_name, new_name = jsonObj[original_name]
        if new_type_name != "disappear" and new_name != "<unk>" and new_type_name != "<unk>":
            print("Renaming " + original_name + " to " + new_name + ".")

            new_type = None

            print(f"Attempting to retype {original_name}/{new_name} to {new_type_name}")

            try:
                ti = find_type_by_name(new_type_name)
                new_type = build_ghidra_type(ti)
                print(f"Changing type of {original_name}/{new_name} to {new_type_name}: {new_type}")
            except Exception as e:
                print(f"Failed to find or build type {new_type_name} exception: {e}")

            try:
                HighFunctionDBUtil.updateDBVariable(var, new_name, new_type, SourceType.USER_DEFINED)
            except Exception as e:
                print(f"Failed to rename/retype {original_name} to {new_name}/{new_type_name} exception: {e}")

        else:
            print("Skipping disappear/unknown variable " + original_name + ".")
    else:
        print("No new name for " + original_name + " in JSON file.")

# Re-decompile
results, high_function = decompile(func)

output['rewritten_decompilation'] = results.getDecompiledFunction().getC()

json.dump(output, open(outputJsonFile, "w"), indent=4)
