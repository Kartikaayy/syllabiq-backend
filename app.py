import os
import json
import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/map": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.route('/map', methods=['POST', 'OPTIONS'])
def map_syllabus():
    if request.method == 'OPTIONS':
        return '', 200

    syllabus_text = ""

    # --- PDF upload ---
    if 'syllabus' in request.files:
        syllabus_file = request.files['syllabus']
        try:
            with pdfplumber.open(syllabus_file) as pdf:
                syllabus_text = " ".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )
        except Exception as e:
            return jsonify({"error": f"Could not read PDF: {str(e)}"}), 400

    # --- Typed topics ---
    typed_topics = request.form.get('topics', '').strip()
    if typed_topics:
        syllabus_text = typed_topics

    if not syllabus_text.strip():
        return jsonify({"error": "Please upload a PDF or type your syllabus topics."}), 400

    # --- Optional context ---
    course_name   = request.form.get('courseName', '').strip()
    course_stream = request.form.get('courseStream', '').strip()
    career_goal   = request.form.get('careerGoal', '').strip()

    context_lines = []
    if course_name:   context_lines.append(f"Course Name: {course_name}")
    if course_stream: context_lines.append(f"Stream / Domain: {course_stream}")
    if career_goal:   context_lines.append(f"Student's Career Goal: {career_goal}")
    context_block = "\n".join(context_lines)
    context_header = ("STUDENT CONTEXT:\n" + context_block) if context_block else ""

    prompt = f"""
You are an expert academic-to-industry skills mapper. Analyze the provided syllabus and map it
to practical real-world skills, project ideas, certifications, and career paths.

Be specific to the actual content. Avoid generic responses.
For certifications, you MUST provide a real, working URL to the official course or certification page.

{context_header}

SYLLABUS CONTENT:
{syllabus_text[:6000]}

Return ONLY valid JSON — no markdown, no explanation, no extra text:
{{
  "skills": [
    {{
      "skill": "Specific skill name",
      "domain": "Industry domain (e.g. Backend Development, Financial Modeling, Circuit Analysis)",
      "level": "Beginner | Intermediate | Advanced"
    }}
  ],
  "project_ideas": [
    {{
      "title": "Project title",
      "description": "1-2 sentences: what the student builds and what it demonstrates",
      "difficulty": "Beginner | Intermediate | Advanced",
      "skills_used": ["skill1", "skill2", "skill3"]
    }}
  ],
  "certifications": [
    {{
      "name": "Full certification name",
      "provider": "Issuing organization (e.g. Google, AWS, Microsoft, Coursera, Udemy)",
      "description": "One sentence on what it validates",
      "relevance": "Short phrase: why this cert aligns with the syllabus",
      "url": "Real direct URL to the cert/course page e.g. https://grow.google/certificates/ or https://aws.amazon.com/certification/"
    }}
  ],
  "career_paths": [
    {{
      "role": "Job title",
      "description": "1-2 sentences: day-to-day responsibilities",
      "demand": "High | Medium | Low",
      "salary_range": "e.g. $65k-$110k / Rs. 6-16 LPA",
      "key_skills": ["skill1", "skill2", "skill3"]
    }}
  ],
  "overall_summary": "3-4 sentences: the syllabus's industry relevance, what roles it prepares students for, and the top 1-2 action items the student should prioritize."
}}

Constraints:
- Return 6-12 skills
- Return 4-6 project ideas (beginner to advanced range)
- Return 3-5 certifications with REAL urls (use known cert pages like coursera.org, aws.amazon.com, cloud.google.com, microsoft.com/certifications, etc.)
- Return 3-5 career paths
- Be specific to the actual syllabus content
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"AI returned invalid JSON: {str(e)}", "raw": text[:300]}), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)