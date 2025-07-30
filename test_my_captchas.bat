@echo off
echo Testing CAPTCHA Imagecho Testing Image 8.png...
python predictor.py --input ./my_captchas/8.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 9.png...
python predictor.py --input ./my_captchas/9.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 10.png...
python predictor.py --input ./my_captchas/10.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo All tests completed!
paused...
echo.

REM Activate virtual environment
call .\.venv\Scripts\activate

REM Test each image from the my_captchas folder using the NEW specialized sssalud model
echo Testing Image 1.png...
python predictor.py --input ./my_captchas/1.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 2.png...
python predictor.py --input ./my_captchas/2.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 3.png...
python predictor.py --input ./my_captchas/3.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 4.png...
python predictor.py --input ./my_captchas/4.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 5.png...
python predictor.py --input ./my_captchas/5.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 6.png...
python predictor.py --input ./my_captchas/6.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 7.png...
python predictor.py --input ./my_captchas/7.png --ckpt ./checkpoints_sssalud/model.pth
echo.

echo Testing Image 8.png...
python predictor.py --input ./my_captchas/8.png --ckpt ./checkpoint/model.pth
echo.

echo Testing Image 9.png...
python predictor.py --input ./my_captchas/9.png --ckpt ./checkpoint/model.pth
echo.

echo Testing Image 10.png...
python predictor.py --input ./my_captchas/10.png --ckpt ./checkpoint/model.pth
echo.

echo All tests completed!
pause
