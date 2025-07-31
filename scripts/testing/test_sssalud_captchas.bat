@echo off
echo Testing CAPTCHA Images with Specialized sssalud Model (20 samples)...
echo.

REM Activate virtual environment
call .\.venv\Scripts\activate

REM Test each image from the my_captchas folder using the specialized sssalud model
echo Testing Image 1.png (Real: UKhGh9)...
python predictor.py --input ./my_captchas/1.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 2.png (Real: 26WanS)...
python predictor.py --input ./my_captchas/2.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 3.png (Real: e4TkHP)...
python predictor.py --input ./my_captchas/3.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 4.png (Real: cGfFE2)...
python predictor.py --input ./my_captchas/4.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 5.png (Real: gnRYZe)...
python predictor.py --input ./my_captchas/5.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 6.png (Real: v76Ebu)...
python predictor.py --input ./my_captchas/6.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 7.png (Real: DUzp49)...
python predictor.py --input ./my_captchas/7.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 8.png (Real: MWR3mw)...
python predictor.py --input ./my_captchas/8.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 9.png (Real: 3h2vUF)...
python predictor.py --input ./my_captchas/9.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 10.png (Real: t2md2m)...
python predictor.py --input ./my_captchas/10.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 11.png (Real: CzSLcN)...
python predictor.py --input ./my_captchas/11.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 12.png (Real: tZyFmr)...
python predictor.py --input ./my_captchas/12.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 13.png (Real: 3N4kv3)...
python predictor.py --input ./my_captchas/13.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 14.png (Real: wvTfzE)...
python predictor.py --input ./my_captchas/14.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 15.png (Real: z56VDI)...
python predictor.py --input ./my_captchas/15.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 16.png (Real: 9mLpn5)...
python predictor.py --input ./my_captchas/16.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 17.png (Real: pKK7BD)...
python predictor.py --input ./my_captchas/17.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 18.png (Real: TpHfBC)...
python predictor.py --input ./my_captchas/18.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 19.png (Real: rpA8pS)...
python predictor.py --input ./my_captchas/19.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 20.png (Real: Z3TUnp)...
python predictor.py --input ./my_captchas/20.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo All 20 tests completed!
pause
