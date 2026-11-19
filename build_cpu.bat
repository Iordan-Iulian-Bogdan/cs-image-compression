@echo off
REM CPU-only build for ImgReconstruct_backend (no ROCm/HIP required).
REM LLVM 23 clang-cl + MS link, OpenCV split:
REM   main exe  -> C:\opencv 4.90 world DLL
REM   cs_tests  -> OpenCV 4.13 static set w/ contrib quality (CV413_ROOT, default
REM                %USERPROFILE%\OneDrive\Documents\opencv\Release)
REM Branch production-cpu: the only build. No HIP, no ONNX Runtime, no
REM restoration nets.
REM Outputs to x64\Cpu so other binaries are untouched.
REM Prereqs: LLVM 23 (winget LLVM.LLVM or portable %LOCALAPPDATA%\llvm-portable),
REM          VS2022 MSVC, OpenCV 4.90 (C:\opencv or portable
REM          %LOCALAPPDATA%\opencv-portable\opencv). No ROCm needed.
REM Usage: build_cpu.bat [main]  -- 'main' builds only ImgReconstruct_backend.exe
REM          and skips cs_tests.exe (which needs the static 4.13 contrib set).
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )
set "LLVM_ROOT=C:\Program Files\LLVM"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" set "LLVM_ROOT=%LOCALAPPDATA%\llvm-portable\clang+llvm-23.1.2-x86_64-pc-windows-msvc"
if not exist "%LLVM_ROOT%\bin\clang-cl.exe" ( echo ERROR: clang-cl not found under "C:\Program Files\LLVM" or "%LOCALAPPDATA%\llvm-portable" & exit /b 1 )
set "PATH=%LLVM_ROOT%\bin;%PATH%"
set "CV490_ROOT=C:\opencv"
if not exist "%CV490_ROOT%\build\include\opencv2\core.hpp" set "CV490_ROOT=%LOCALAPPDATA%\opencv-portable\opencv"
if not exist "%CV490_ROOT%\build\include\opencv2\core.hpp" ( echo ERROR: OpenCV 4.90 not found under "C:\opencv" or "%LOCALAPPDATA%\opencv-portable\opencv" & exit /b 1 )
set "ROOT=%~dp0"
set "OUT=%ROOT%x64\Cpu"
set "OBJ=%OUT%\obj"
set "OBJT=%OUT%\obj_test"
mkdir "%OUT%" 2>nul
mkdir "%OBJ%" 2>nul
mkdir "%OBJT%" 2>nul
set "CV490_INC=/I %CV490_ROOT%\build\include"
if not defined CV413_ROOT set "CV413_ROOT=%USERPROFILE%\OneDrive\Documents\opencv\Release"
set "CV413_INC=/I %CV413_ROOT%\include"
set "DEFS=/DNDEBUG /D_CONSOLE /D_CRT_SECURE_NO_WARNINGS"
set "OPTS=/c /O2 /fp:fast /arch:AVX512 /clang:-mavx512f /clang:-mavx512bw /clang:-mavx512dq /clang:-mavx512vl /clang:-mavx512cd /Oi /std:c++17 /EHsc /MD /openmp /Zi /clang:-Wunused-function /clang:-Wunused-variable /clang:-Wunused-parameter"
set "BE_INC=/I %ROOT%ImgReconstruct_backend"

echo ============ main objects [opencv 4.90 world] ============
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\cs_sampler.cpp" /Fo"%OBJ%\cs_sampler.obj"
if errorlevel 1 ( echo ERROR cs_sampler & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\container_format.cpp" /Fo"%OBJ%\container_format.obj"
if errorlevel 1 ( echo ERROR container_format & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\sample_coder.cpp" /Fo"%OBJ%\sample_coder.obj"
if errorlevel 1 ( echo ERROR sample_coder & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\helper_functions.cpp" /Fo"%OBJ%\helper_functions.obj"
if errorlevel 1 ( echo ERROR helper_functions & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\superres.cpp" /Fo"%OBJ%\superres.obj"
if errorlevel 1 ( echo ERROR superres & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_decoder.cpp" /Fo"%OBJ%\image_decoder.obj"
if errorlevel 1 ( echo ERROR image_decoder & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_encoder.cpp" /Fo"%OBJ%\image_encoder.obj"
if errorlevel 1 ( echo ERROR image_encoder & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_tiles.cpp" /Fo"%OBJ%\image_tiles.obj"
if errorlevel 1 ( echo ERROR image_tiles & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_preview.cpp" /Fo"%OBJ%\image_preview.obj"
if errorlevel 1 ( echo ERROR image_preview & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV490_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\ImgReconstruct_backend.cpp" /Fo"%OBJ%\main.obj"
if errorlevel 1 ( echo ERROR main & exit /b 1 )

if "%~1"=="main" goto :linkmain
echo ============ test objects [opencv 4.13 static + contrib] ============
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\cs_sampler.cpp" /Fo"%OBJT%\cs_sampler.obj"
if errorlevel 1 ( echo ERROR cs_sampler_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\container_format.cpp" /Fo"%OBJT%\container_format.obj"
if errorlevel 1 ( echo ERROR container_format_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\sample_coder.cpp" /Fo"%OBJT%\sample_coder.obj"
if errorlevel 1 ( echo ERROR sample_coder_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\helper_functions.cpp" /Fo"%OBJT%\helper_functions.obj"
if errorlevel 1 ( echo ERROR helper_functions_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\superres.cpp" /Fo"%OBJT%\superres.obj"
if errorlevel 1 ( echo ERROR superres_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_decoder.cpp" /Fo"%OBJT%\image_decoder.obj"
if errorlevel 1 ( echo ERROR image_decoder_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_encoder.cpp" /Fo"%OBJT%\image_encoder.obj"
if errorlevel 1 ( echo ERROR image_encoder_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_tiles.cpp" /Fo"%OBJT%\image_tiles.obj"
if errorlevel 1 ( echo ERROR image_tiles_t & exit /b 1 )
clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%ImgReconstruct_backend\image_preview.cpp" /Fo"%OBJT%\image_preview.obj"
if errorlevel 1 ( echo ERROR image_preview_t & exit /b 1 )
REM test sources: one object per tests\<name>.cpp, listed once here
set "TEST_UNITS=test_main test_primitives test_solvers test_modes test_photo_sr"
for %%T in (%TEST_UNITS%) do (
    clang-cl %OPTS% %DEFS% %CV413_INC% %BE_INC% "%ROOT%tests\%%T.cpp" /Fo"%OBJT%\%%T.obj"
    if errorlevel 1 ( echo ERROR %%T & exit /b 1 )
)

:linkmain
set "SYSLIBS=libomp.lib user32.lib gdi32.lib kernel32.lib ole32.lib oleaut32.lib uuid.lib advapi32.lib shell32.lib winspool.lib comdlg32.lib"

echo [link] ImgReconstruct_backend.exe
link /OUT:"%OUT%\ImgReconstruct_backend.exe" /DEBUG /SUBSYSTEM:CONSOLE /LIBPATH:%CV490_ROOT%\build\x64\vc16\lib "/LIBPATH:%LLVM_ROOT%\lib" "%OBJ%\main.obj" "%OBJ%\cs_sampler.obj" "%OBJ%\container_format.obj" "%OBJ%\helper_functions.obj" "%OBJ%\sample_coder.obj" "%OBJ%\superres.obj" "%OBJ%\image_decoder.obj" "%OBJ%\image_encoder.obj" "%OBJ%\image_tiles.obj" "%OBJ%\image_preview.obj" opencv_world490.lib %SYSLIBS%
if errorlevel 1 ( echo ERROR linking exe & exit /b 1 )
if "%~1"=="main" goto :finish

echo [link] cs_tests.exe (static 4.13 set)
set "STATIC413=opencv_core4130.lib opencv_imgproc4130.lib opencv_highgui4130.lib opencv_imgcodecs4130.lib opencv_dnn4130.lib opencv_dnn_superres4130.lib opencv_photo4130.lib opencv_videoio4130.lib opencv_quality4130.lib ippicvmt.lib ippiw.lib ipphal.lib ittnotify.lib libprotobuf.lib libjpeg-turbo.lib libpng.lib libtiff.lib libopenjp2.lib IlmImf.lib zlib.lib ade.lib"
link /OUT:"%OUT%\cs_tests.exe" /DEBUG /SUBSYSTEM:CONSOLE "/LIBPATH:%CV413_ROOT%\lib" "/LIBPATH:%LLVM_ROOT%\lib" "%OBJT%\test_main.obj" "%OBJT%\test_primitives.obj" "%OBJT%\test_solvers.obj" "%OBJT%\test_modes.obj" "%OBJT%\test_photo_sr.obj" "%OBJT%\cs_sampler.obj" "%OBJT%\container_format.obj" "%OBJT%\helper_functions.obj" "%OBJT%\sample_coder.obj" "%OBJT%\superres.obj" "%OBJT%\image_decoder.obj" "%OBJT%\image_encoder.obj" "%OBJT%\image_tiles.obj" "%OBJT%\image_preview.obj" %STATIC413% %SYSLIBS%
if errorlevel 1 ( echo ERROR linking tests & exit /b 1 )

:finish
copy "%LLVM_ROOT%\bin\libomp.dll" "%OUT%\" >nul
copy "%CV490_ROOT%\build\x64\vc16\bin\opencv_world490.dll" "%OUT%\" >nul
echo BUILD OK: %OUT%
