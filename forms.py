from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
# index form
class StartForm(FlaskForm):
     start= SubmitField('开始检测')
