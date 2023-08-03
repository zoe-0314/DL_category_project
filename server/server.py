from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import model
import pymysql

app = Flask(__name__)

@app.route('/file_upload', methods=['GET', 'POST']) # 함수 이름이 아닌 해당 코드로 html과 연결됨 
def file_upload():
    save_f = []     # save file name list
    files_name = []
    if request.method == 'POST':
        up_f = request.files.getlist("file[]")  # request: html에서 보낸 매개변수를 받아서 저장, html의 name="file[]"과 이름이 같아야함
        for f in up_f:
            files_name.append(secure_filename(f.filename))
            f_name = './static/IMG/' + secure_filename(f.filename)
            f.save(f_name)
            save_f.append(f_name)
        label = inference(save_f)
        return render_template('file_upload.html', name=save_f, label = label, files_name = ','.join(files_name))    # 지정해준 파일로 가서 열린다. 
        
    return render_template('file_upload.html')

def inference(filename):
    # 상세 분류
    label =  model.get_best_class(filename)
    return label

@app.route('/db_upload', methods=['GET', 'POST']) # 함수 이름이 아닌 해당 코드로 html과 연결됨 
def db_upload():
    if request.method == 'POST':
        label = request.form["check_category"]
        label = [l.strip() for l in label.split(',')]
        filename = request.form["filename"]
        
        db_config = {
            'host': 'localhost',
            'user': 'young',
            'password': 'password',
            'database': 'dlClothes',
            'cursorclass': pymysql.cursors.DictCursor
        }

        try:
            connection = pymysql.connect(**db_config)
            cursor = connection.cursor()

            # 소매 기장 없는 경우
            if label[3] == 'None':
                query = "INSERT INTO product (category1, category2, material, filename) VALUES (%s, %s, %s, %s)"
                cursor.execute(query, (label[0], label[1], label[2], filename))
            else:
                query = "INSERT INTO product (category1, category2, material, sleeveLength, filename) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(query, (label[0], label[1], label[2], label[3], filename))
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except pymysql.Error as err:
            print(err)
            return render_template('file_upload.html')    # 초기화면
    return render_template('file_upload.html')    # 저장 후 초기 화면으로 돌아가기 

if __name__ == '__main__':
    app.run('0.0.0.0', port=8765, debug=True)