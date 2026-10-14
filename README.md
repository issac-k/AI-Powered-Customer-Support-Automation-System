🎧 AI-Powered Customer Support Automation System
A Streamlit app that reads a customer complaint, uses the Groq LLM API to classify it, detect its priority, and write a draft reply, then lets a human review and approve the reply before it is used.
Features
AI ticket analysis: category, sentiment, priority and a professional draft response
Human in the loop: every draft is saved as Pending Review; an agent edits it, then approves or rejects it
Ticket dashboard: metrics (total, pending, approved, rejected, urgent) and filters by status, priority and category
Urgent-first sorting: the most critical tickets appear at the top
Safe AI handling: validated labels, JSON clean-up, automatic fallback model, friendly error messages, and protection against prompt injection
Local storage: tickets are saved in a SQLite database (`support_tickets.db`)
> Approving a response only saves it. Nothing is emailed to the customer.
Tech Stack
Part	Tool
UI	Streamlit
AI	Groq API (`openai/gpt-oss-20b`, fallback `llama-3.3-70b-versatile`)
Database	SQLite
Language	Python 3.12+
Run Locally (Windows)
Get a free API key at https://console.groq.com/keys
Clone the project
```
   git clone https://github.com/issac-k/AI-Powered-Customer-Support-Automation-System.git
   cd AI-Powered-Customer-Support-Automation-System
   ```
Create and activate a virtual environment
```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
Install the packages
```
   pip install -r requirements.txt
   ```
Add your key. Create a file named `.env` in the project folder:
```
   GROQ_API_KEY=your_key_here
   ```
Optional: `GROQ_MODEL=openai/gpt-oss-20b`
Start the app
```
   streamlit run app.py
   ```
It opens at http://localhost:8501
How to Use
Open the Create Support Ticket tab, enter the customer's name and message, and click Analyze Ticket with AI.
Open the Ticket Dashboard tab, expand a ticket, edit the draft if needed, then click Approve Response or Reject Draft.
Deploy
GitHub Pages cannot run Python apps. Deploy this app free on Streamlit Community Cloud:
Choose this repository, branch `main`, main file `app.py`.
In Advanced settings → Secrets, add:
```
   GROQ_API_KEY = "your_key_here"
   ```
Click Deploy.
Note: on Streamlit Cloud the SQLite file is temporary, so tickets reset when the app restarts.
Project Structure
```
app.py              # the whole application
requirements.txt    # dependencies
.gitignore          # keeps .env, .venv and the database out of Git
```
Security
Never commit your `.env` file or API key.
If a key is ever leaked, delete it in the Groq console and create a new one.
