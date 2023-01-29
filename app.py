import os
import uuid
from flask import Flask, render_template, redirect, url_for, request, jsonify
from forms import StartForm
from werkzeug.utils import secure_filename
import json
from MyDetect import Detect

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'secret string')
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# 验证是否是图片文件
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# 文件重新命名
def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename
@app.route('/', methods=['GET', 'POST'])
def index():
    form = StartForm()
    if form.validate_on_submit():
        return redirect(url_for('detect'))
    return render_template('index.html', form=form)

@app.route('/upload_img',methods=['post'])
def upload_img():
    if 'img' not in request.files:
        resp = jsonify({'message':'没有上传图片'})
        resp.status_code = 400
        return resp
    imagelist = request.files.getlist('img')
    image = imagelist[0]
    print(image)
    errors={}
    dic={}
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        filename = random_filename(filename)
        imgPath = app.config['UPLOAD_FOLDER']+filename
        image.save(imgPath)
        imgPath_for_js = '../'+imgPath
        imgPath_for_js = 'url('+imgPath_for_js+')'
        success = True
    else:
        print("error")
        errors['err_message'] = '仅支持jpg、jpeg、png'
        resp = jsonify(errors)
        resp.status_code = 404
        return resp

    if success:
        dic['suc_message'] = '成功加载图片'
        dic['imgPath_for_js']=imgPath_for_js
        dic['imgPath'] = imgPath
        dic['filename']=filename
        resp = jsonify(dic)
        with open('static/json/data.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)
        resp.status_code = 201
        return resp

@app.route('/detect_img',methods=['post'])
def detect_img():
    dic = json.load(open('static/json/data.json','r'))
    detImg = dic['imgPath']
    print(detImg)
    mydet = Detect(source=detImg,name='detected',project='static',conf_thres=0.4)
    info = mydet.detect()
    detectedImgPath = "static/detected/"+dic['filename']
    print(detectedImgPath)
    dic['detectedImgPath']=detectedImgPath
    detectedImgPath_for_js = '../' + detectedImgPath
    detectedImgPath_for_js = 'url(' + detectedImgPath + ')'
    dic['detectedImgPath_for_js'] = detectedImgPath_for_js
    detailInfo = info;
    dic['detailInfo'] = detailInfo
    print(detailInfo)
    resp = jsonify(dic)
    with open('static/json/data.json', 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)
    return resp

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    return render_template('detect.html')


if __name__ == '__main__':
    app.run(debug=True)
# tensorboard --logdir=runs/train