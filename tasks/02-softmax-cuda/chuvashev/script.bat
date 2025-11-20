nvcc ^
	src/kernel.cu src/main.cpp src/utils.cpp src/simd_utils.cpp ^
	-o output.exe ^
	-ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64" ^
	--machine 64 ^
	-std=c++17 ^
	--generate-code=arch=compute_75,code=[compute_75,sm_75] ^
	-Xcompiler="/EHsc -Ob2" ^
	-Xcompiler "/EHsc /W1 /nologo /O2 /FS /MD" ^
	-DTHREADS_PER_BLOCK=%1 ^
	-Iinclude/
	

