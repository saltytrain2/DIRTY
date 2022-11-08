from ghidra.app.decompiler import DecompInterface
import os


def main():
    print(os.getcwd())
    decomp = DecompInterface()
    decomp.openProgram(currentProgram)

    for function in currentProgram.getListing().getFunctions(True):
        if function.getName() != 'main':
            continue

        decomp_results = decomp.decompileFunction(function, 30, None)
        
        if decomp_results.decompileCompleted():
            fn_sig = decomp_results.getDecompiledFunction().getSignature()
            fn_code = decomp_results.getDecompiledFunction().getC()
            print(f"Function: {fn_sig}")
            print(fn_code)
            #tokens = []
            #for token in decomp_results.getCCodeMarkup():
            #    tokens.append(token.toString())
            #print(tokens)

            with open('file.txt', 'w') as file:
                file.write(fn_code)

        else:
            print("Error in decompilation")


if __name__ == '__main__':
    main()