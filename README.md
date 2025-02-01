**# SimpleChatbot_R1Deepseek_1.5B
 is a web-based AI application using deepseek-r1:1.5b that helps users write, understand, and debug code by displaying well-formatted code snippets with syntax highlighting.  

This application is designed for developers, students, or anyone who needs programming assistance. With a responsive interface and powerful features, users can quickly get answers and ready-to-use code.  

---

## 🚀 Features  

✅ **Markdown-style Code Blocks** – Supports triple backticks (```) for displaying code snippets.  
✅ **Syntax Highlighting** – Highlights syntax for Python, JavaScript, and CSS using Prism.js.  
✅ **📋 Copy-to-Clipboard** – One-click button to copy code easily.  
✅ **📱 Responsive Design** – Works smoothly on desktop & mobile.  
✅ **🎤 Voice Input Support** – Allows users to input queries via voice.  
✅ **⌨️ Keyboard Shortcuts** –  
&nbsp;&nbsp;&nbsp;&nbsp;🔹 `/` – Focus on input field.  
&nbsp;&nbsp;&nbsp;&nbsp;🔹 `Ctrl+Enter` or `Cmd+Enter` – Send message.  
✅ **📏 Auto-expanding Input Area** – Input box resizes automatically.  
✅ **🗑️ Clear Chat** – Clear conversation with one click.  

---

## 🛠️ Technologies  

### 🌐 Frontend  
- 🏗 **HTML, CSS, JavaScript**  
- 🎨 **Tailwind CSS**  
- 🎨 **Prism.js** – Syntax highlighting  
- 🎙 **Web Speech API** – Voice input  

### 🖥 Backend  
- 🐍 **Flask (Python)**  
- 🧠 **Ollama API** – Local Deepseek Model  

---

## 📦 Installation  

### 🔻 Clone Repository  
```bash
git clone https://github.com/username/repo-name.git
cd repo-name
```
### 📌 Install Dependencies
Ensure Python and Flask are installed, then run:
```bash
pip install -r requirements.txt
```
### 🚀 Run the Application
```bash
python app.py
```
### 🚧 Development Status
- ⚠️ This project is still under development. Some features are incomplete, and bugs may exist. If you find an issue, please open an Issue or submit a Pull Request.

### 🐛 Known Issues
- ❌ Voice Input Bug – May not work on some browsers.
- ❌ Limited Syntax Highlighting – Supports Python, JavaScript, CSS only.
- ❌ Responsiveness Issues – Some UI elements may not scale properly.
