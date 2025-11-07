from flask import Flask,request,render_template, redirect, url_for, session, jsonify
from functools import wraps
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from werkzeug.security import generate_password_hash, check_password_hash

# Add missing imports required by this file
import os
import sqlite3

application=Flask(__name__, template_folder="templates", static_folder="static")

# secret key for sessions (override via env var in production)
application.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

app=application

DB_PATH = os.path.join(os.path.dirname(__file__), 'portal.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # users table for staff accounts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT UNIQUE,
        profession TEXT,
        password_hash TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        student_id TEXT,
        gender TEXT,
        year INTEGER,
        reading_score INTEGER,
        writing_score INTEGER,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username, email, profession, password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "email": row[2], "profession": row[3], "password_hash": row[4]}

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user'):
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email = request.form.get('email','').strip().lower()
        profession = request.form.get('profession','').strip()
        password = request.form.get('password','')
        password2 = request.form.get('password2','')
        if not username or not email or not password:
            return render_template('signup.html', error="All fields are required.")
        if password != password2:
            return render_template('signup.html', error="Passwords do not match.")
        if get_user_by_email(email):
            return render_template('signup.html', error="An account with that email already exists.")
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, email, profession, password_hash, created_at) VALUES (?,?,?,?,?)",
                    (username, email, profession, password_hash, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        # auto-login after signup
        session['user'] = email
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('signup.html')

# make login available from navbar form and full page
@app.route('/login', methods=['GET','POST'])
def login():
    # support both navbar form (POST) and full page
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        user = get_user_by_email(email)
        if user and check_password_hash(user['password_hash'], password):
            session['user'] = user['email']
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials.")
    # GET
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('username', None)
    return redirect(url_for('index'))

# Replace the existing empty index route with this implementation
# (shows dashboard when signed in, otherwise shows the public home page)
@app.route('/')
def index():
    # if signed in go to dashboard otherwise show public home page
    if session.get('user'):
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get student data for dashboard
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, student_id, gender, year, reading_score, writing_score FROM students ORDER BY created_at DESC")
    students = [
        {
            "id": row[0],
            "name": row[1],
            "student_id": row[2],
            "gender": row[3],
            "year": row[4],
            "reading_score": row[5],
            "writing_score": row[6]
        }
        for row in cur.fetchall()
    ]
    
    # Get statistics for dashboard cards
    cur.execute("SELECT COUNT(*) FROM students")
    total_students = cur.fetchone()[0]
    
    cur.execute("SELECT AVG(reading_score), AVG(writing_score) FROM students")
    avg_scores = cur.fetchone()
    avg_reading = round(avg_scores[0] or 0, 1)
    avg_writing = round(avg_scores[1] or 0, 1)
    
    conn.close()

    return render_template('portal.html',
                         user=session.get('user'),
                         username=session.get('username'),
                         students=students,
                         stats={
                             "total_students": total_students,
                             "avg_reading": avg_reading,
                             "avg_writing": avg_writing,
                             "total_records": len(students)
                         })

@app.route('/add_student', methods=['POST'])
@login_required
def add_student():
    try:
        # accept both form POST and JSON
        if request.is_json:
            payload = request.get_json()
            name = payload.get('name','').strip()
            student_id = payload.get('student_id','').strip()
            gender = payload.get('gender','')
            year = int(payload.get('year', datetime.now().year))
            reading_score = int(float(payload.get('reading_score', 0)))
            writing_score = int(float(payload.get('writing_score', 0)))
        else:
            name = request.form.get('name','').strip()
            student_id = request.form.get('student_id','').strip()
            gender = request.form.get('gender','')
            year = int(request.form.get('year', datetime.now().year))
            reading_score = int(float(request.form.get('reading_score', 0)))
            writing_score = int(float(request.form.get('writing_score', 0)))
        created_at = datetime.utcnow().isoformat()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO students (name, student_id, gender, year, reading_score, writing_score, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (name, student_id, gender, year, reading_score, writing_score, created_at))
        conn.commit()
        conn.close()
        if request.is_json:
            return jsonify({"status":"ok"}), 201
        return redirect(url_for('dashboard'))
    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        return render_template('portal.html', error=str(e))

@app.route('/api/average_by_year')
@login_required
def api_average_by_year():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT year, AVG(reading_score), AVG(writing_score)
        FROM students
        GROUP BY year
        ORDER BY year
    """)
    rows = cur.fetchall()
    conn.close()
    data = {"years": [], "avg_reading": [], "avg_writing": []}
    for r in rows:
        data["years"].append(r[0])
        data["avg_reading"].append(round(r[1] or 0,2))
        data["avg_writing"].append(round(r[2] or 0,2))
    return jsonify(data)

@app.route('/api/student_history')
@login_required
def api_student_history():
    student_id = request.args.get('student_id')
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT year, reading_score, writing_score, created_at
        FROM students
        WHERE student_id = ?
        ORDER BY year
    """, (student_id,))
    rows = cur.fetchall()
    conn.close()
    data = {"records": []}
    for r in rows:
        data["records"].append({"year": r[0], "reading": r[1], "writing": r[2], "created_at": r[3]})
    return jsonify(data)

# new API to compare multiple students by id (returns averages per selected students)
@app.route('/api/compare_students')
@login_required
def api_compare_students():
    ids = request.args.getlist('student_id')
    if not ids:
        return jsonify({"error":"student_id parameters required, e.g. ?student_id=ID1&student_id=ID2"}), 400
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cur.execute(f"""
        SELECT student_id, name, AVG(reading_score) as avg_reading, AVG(writing_score) as avg_writing
        FROM students
        WHERE student_id IN ({placeholders})
        GROUP BY student_id, name
    """, ids)
    rows = cur.fetchall()
    conn.close()
    data = [{"student_id": r[0], "name": r[1], "avg_reading": round(r[2] or 0,2), "avg_writing": round(r[3] or 0,2)} for r in rows]
    return jsonify({"results": data})

# keep the existing prediction route and functionality
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            data=CustomData(
                gender=request.form.get('gender', ''),
                race_ethnicity=request.form.get('ethnicity') or request.form.get('race_ethnicity', ''),
                parental_level_of_education=request.form.get('parental_level_of_education', ''),
                lunch=request.form.get('lunch', ''),
                test_preparation_course=request.form.get('test_preparation_course', ''),
                reading_score=int(float(request.form.get('reading_score', 0))),
                writing_score=int(float(request.form.get('writing_score', 0))),
            )
            pred_df=data.get_data_as_data_frame()

            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(pred_df)
            return render_template('home.html',results=results[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('home.html', results=f"Error: {str(e)}")
    

# provide current time helper to templates (used in portal add form)
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow}
    
if __name__=='__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)