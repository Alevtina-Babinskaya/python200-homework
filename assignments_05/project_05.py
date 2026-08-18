from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_role = """You are a job application coach. You help job seekers improve their resumes, cover letters, LinkedIn profiles, and other job application materials. 
Provide clear, constructive, and encouraging feedback, and suggest ways to improve clarity, organization, professionalism, and relevance to the job. 
Stay focused on job application materials. If information is missing, ask questions before making assumptions. 
Do not invent qualifications, work experience, education, skills, or achievements that the user has not provided.
Always remind the user to carefully review and edit your suggestions before submitting any application materials. 
Acknowledge that you may not know the specific expectations or industry norms for every employer or field, and encourage the user to use their own judgment 
when deciding whether to apply your recommendations."""
# I asked model not to invent qualifications because I've already dealt with AI resume reviewers before. They often invent skills and qualification 
# that didn't exist in original version.
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.
    Return ONLY a valid JSON. Do not use markdowns, explanations, and any other additional text and signs. Do NOT wrap the list in an object.
    Do NOT use a "bullets" key. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """
    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        print("The response is not valid JSON.")
    

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
data = rewrite_bullets(bullets)
print(data)
print(type(data))
for item in data:
    print(f"Original bullet: {item['original']}")
    print(f"Improved bullet: {item['improved']}")


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    return response
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
data = generate_cover_letter(job_title, background)
print(data)

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("The text hasn't passed moderation check")
        print("Please, rephrase your message")
        return False
    else:
        return True
print(is_safe("I've killed people before and I kill you."))
print(is_safe("I had worked as interface designer before moving to the US"))
borderline_text = "You are no more but a filthy mudblood"
result = client.moderations.create(
    model="omni-moderation-latest",
    input=borderline_text
)

print(result.results[0].categories)

def run_chatbot():
    messages = [
        {"role": "system", "content": system_role}
        ]
    
    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType quit at any time to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break
        if not user_input:
            continue
        if not is_safe(user_input):
            continue
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            response = rewrite_bullets(raw_bullets)
            print(response)
            messages.append({"role": "user", "content": user_input + "\n" + "\n".join(raw_bullets)})
            messages.append({"role": "assistant", "content": response})
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            messages.append({"role": "user", "content": user_input})
            response = generate_cover_letter(job_title, background)
            print(response)
            messages.append({"role": "assistant", "content": response})
        else:
            messages.append({"role": "user", "content": user_input+ "\nJob title: " + job_title + "\nBackground: " + background})
            response = get_completion(messages)
            print(response)
            messages.append({"role": "assistant", "content": response})
            pass
if __name__ == "__main__":
    run_chatbot()

# 1. If the bot is trained mostly on text written by certain groups of people, its advice may reflect their communication style and expectations. 
# Some direct communication style may not be appropriate for every culture or profession. Users should adapt the suggestions to their own field and background.

# 2. Chatbot responses may sound unnatural, and an employer may recognize AI-generated writing. 
# Also, chatbots sometimes invent details or exaggerate achievements, so submitting the output without reviewing it could result in inaccurate or misleading information.

# 3. I would add a clear warning telling users to review and personalize the bot’s output before submitting it. 
# This would remind users that the bot’s response may contain mistakes or information that does not accurately represent them.
