#!/usr/bin/env python
# coding: utf-8

from typing import Optional
from flask import Flask, request, render_template
from fastai.vision import *
from io import BytesIO
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, StringField
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from io import BytesIO
from torchvision import transforms
import base64

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gX6CzYrxAiMpMFDIobwGBufQ9VvyJOCy'
Bootstrap(app)

app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'assets')

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

model = load_learner('skin_400')

def colorBytes(fpath):
    img = open_image(fpath)
    preds = model.predict(img)
    img_io = BytesIO()
    transforms.ToPILImage()(preds[0].data).save(img_io,'png')
    img_io.seek(0)
    return img_io

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'File type not supported'), FileRequired('Empty file')])
    url = StringField('Image URL', validators=[])
    submit = SubmitField('Upload')

@app.route('/', methods=['GET', 'POST'])
def main():
    form = UploadForm()
    for f in os.listdir(app.config['UPLOADED_PHOTOS_DEST']):
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'],f))

    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'],filename)
        img = colorBytes(path)
        data = img.getvalue()
        byte64image = base64.b64encode(data)
        byte64image = byte64image.decode()

    else:
        file_url = None
        byte64image = None

    return render_template('index.html', form=form, file_url=file_url, byte64image=byte64image, upload=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
