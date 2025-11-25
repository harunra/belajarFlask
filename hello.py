
from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

class ContactForm(FlaskForm):
    name = StringField('Nama', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    submit = SubmitField('Kirim')
    born = StringField('Tanggal Lahir')
    school = StringField('Asal Sekolah')
    cellphone = StringField('No. Handphone')




@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        # Proses data
        return redirect(url_for('thank_you'))
    return render_template('contact.html', form=form)

@app.route('/pmb', methods=['GET', 'POST'])
def pmb():
    form = PmbForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        born = form.born.data
        school = form.school.data
        cellphone = form.cellphone.data
        # Proses data
        return redirect(url_for('thank_you'))
    return render_template('contact.html', form=form)

@app.route('/thank-you')
def thank_you():
    return "Terima kasih telah mengirimkan form!"

if __name__ == '__main__':
    app.run(debug=True)
