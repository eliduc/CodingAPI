# CodingAPI

**Version 9.1 (May 20, 2025)**

CodingAPI is a comprehensive Python application designed for generating, auditing, and correcting code using multiple Large Language Model (LLM) providers. It features a robust architecture, enhanced error handling, security measures, performance optimizations, and an intuitive graphical user interface (GUI).

## Functionality Description

CodingAPI offers a rich set of features to streamline the code development lifecycle with the assistance of AI:

* **Multi-LLM Support:**
    * Integrates with various LLM providers, including OpenAI (GPT-4o, GPT-4.1, o3, o4-mini), Anthropic (Claude 3.7 Sonnet), Google (Gemini 2.5 Pro, Gemini 2.0 Flash), and DeepSeek (DeepSeek R1).
    * Dynamically discovers and allows selection of available models from configured providers.
    * Optimized parameter settings for different model families and task types (web app, data science, algorithm).
* **Code Generation & Correction Modes:**
    * **Multiple Correction Mode:** Generates initial code using a selected LLM and then iteratively audits and corrects it using another (or the same) LLM for a specified number of iterations.
    * **Multiple Creation Mode:** Generates code using multiple selected LLMs simultaneously, allowing users to compare and choose the best result.
* **Advanced Prompting & Output:**
    * Utilizes sophisticated prompt templates tailored for code generation, auditing, and correction.
    * Supports a delimiter-based format (`###CODE###`, `###CORRECTIONS###`) for reliable code and correction exchange between LLMs, improving parsing reliability.
* **User Interface (GUI):**
    * Built with Tkinter, providing an intuitive interface for:
        * Entering program descriptions and project names.
        * Selecting programming languages.
        * Choosing LLMs for coding and auditing.
        * Specifying the number of correction iterations.
        * Uploading existing code files for modification.
        * Viewing generated code and audit results side-by-side.
        * Monitoring the progress of generation and correction cycles.
    * Includes tooltips for all UI elements for better usability.
    * "Run app" button to execute generated Python applications or view terminal output files (`.trm`).
* **Configuration & Management:**
    * Unified Settings dialog (cog icon) for:
        * Configuring the output directory for generated files.
        * Managing API Keys (supports environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, or an `APIKeys` file).
        * Selecting active LLM models for coding and auditing.
    * API key validation and model availability checking.
* **Data Persistence & Management:**
    * Stores all generated code, audit reports, and project information in an SQLite database (`codingapi.db` located in a `/data` subdirectory relative to the application's execution path).
    * Tracks coding projects with timestamps, modes, LLMs used, and iteration details.
    * Configuration files (settings, model selections) are also stored in the `/data` subdirectory for better portability.
* **Security & Error Handling:**
    * Secure credential management for API keys (uses `keyring` if available, with Fernet encryption as a fallback).
    * Input validation and sanitization.
* **Performance:**
    * Asynchronous processing capabilities using `ThreadPoolExecutor` for concurrent operations.
    * Caching for LLM responses to avoid redundant API calls using `diskcache`.

## Installation

1.  **Prerequisites:**
    * Python 3.7+
    * Ensure `pip` is installed.

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/CodingAPI.git](https://github.com/your-username/CodingAPI.git)
    cd CodingAPI
    ```
    (Replace `your-username` with the actual repository path).

3.  **Install Dependencies:**
    CodingAPI uses several Python libraries. Install them using pip:
    ```bash
    pip install openai anthropic google-generativeai requests configparser keyring cryptography tenacity rich diskcache psutil matplotlib numba torch torchvision torchaudio
    ```
    * `openai`: For OpenAI and DeepSeek models.
    * `anthropic`: For Claude models.
    * `google-generativeai`: For Gemini models.
    * `requests`: For HTTP requests.
    * `configparser`: For legacy API key file reading.
    * `keyring`, `cryptography`: For secure API key storage (optional but recommended).
    * `tenacity`: For retry mechanisms (though its direct use isn't prominent in the provided `CodingAPI.py` snippet, it's listed as an import).
    * `rich`: For enhanced terminal output (optional, for CLI).
    * `diskcache`: For caching LLM responses (optional but recommended).
    * `psutil`: Used by `Connect4-AlphaZero.py` for process management during shutdown.
    * `matplotlib`, `numba`, `torch`, `torchvision`, `torchaudio`: Required by the `Connect4-AlphaZero.py` example.

4.  **Configure API Keys:**
    * **Environment Variables (Recommended):** Set the following environment variables with your API keys:
        * `OPENAI_API_KEY`
        * `ANTHROPIC_API_KEY`
        * `GEMINI_API_KEY`
        * `DEEPSEEK_API_KEY`
    * **APIKeys File (Legacy):** Alternatively, create a file named `APIKeys` in the application's root directory (or in the `/data` subdirectory if preferred, though the primary lookup is root) with the following format:
        ```ini
        [API Keys]
        OpenAI = YOUR_OPENAI_KEY
        Claude = YOUR_ANTHROPIC_KEY
        Gemini = YOUR_GEMINI_KEY
        DeepSeek = YOUR_DEEPSEEK_KEY
        ```
    * **In-App Configuration:** You can also manage API keys and model selections via the "Settings" (⚙️) dialog within the application. The app will guide you if keys are missing. API keys are stored securely.

5.  **Run the Application:**
    ```bash
    python CodingAPI.py
    ```
    This will launch the graphical user interface. A `data` subdirectory will be created in the current working directory if it doesn't exist, to store configuration and the database.

## Usage Examples

### Example 1: Generating a Simple Python Script

1.  **Launch CodingAPI.**
2.  In the "Project name" field, enter `MyFirstScript`. This name will be used for the output directory (e.g., `data/MyFirstScript/`) and database entries.
3.  In the "Program Description/Comment" field, type:
    `Create a Python script that asks the user for their name and then prints a greeting message.`
4.  Select "Python" from the "Programming Language" dropdown.
5.  **Mode Selection:**
    * **Multiple Correction Mode (Default):**
        * Choose a "Coding LLM" (e.g., "OpenAI GPT4o").
        * Choose an "Auditing LLM" (e.g., "Claude 3.7 Sonnet").
        * Set "Iterations" (e.g., 3).
    * **Multiple Creation Mode:**
        * Click the "Multiple Creation" radio button.
        * In the "Coding LLM" listbox that appears, select multiple LLMs (e.g., "OpenAI GPT4o", "Gemini 2.5 Pro", "Claude 3.7 Sonnet").
6.  Click "Start Coding".
7.  **Process:**
    * **Multiple Correction:** The application will first generate the code using the coding LLM. Then, it will iterate: the auditing LLM will review the code, and the coding LLM will attempt to fix any identified issues. The "Current Code" and "Audit Result" panes will update with each step.
    * **Multiple Creation:** The application will generate code using each selected LLM. The "Code Creation Process" pane will show the status for each model. You can click on a completed model in that pane to view its generated code in the "Current Code" pane.
8.  **Output:**
    * Generated code files and audit logs will be saved in a directory named after your project (e.g., `MyFirstScript/`) inside the configured output directory (default is a subdirectory with the project name within the application's execution directory, or within the `/data` folder if no custom path is set).
    * The final code and all intermediate steps will also be stored in the `codingapi.db` database (located in `data/codingapi.db`).

### Example 2: AI for Connect4 Game (AlphaZero Algorithm)

This repository includes an example of a more complex application that can be developed and refined using CodingAPI: an AI that learns to play Connect4 using an AlphaZero-style algorithm.

* **Location:** The source code for this example, `Connect4-AlphaZero.py`, is located in the `Connect4/` subdirectory.
* **Functionality:** This Python script implements:
    * A Connect4 game environment.
    * A Convolutional Neural Network (CNN) or a Residual Network (ResNet) to predict move policies and game values.
    * Monte Carlo Tree Search (MCTS) enhanced by the neural network (PUCT algorithm).
    * A self-play loop where the AI plays against itself to generate training data.
    * A training loop to update the neural network based on self-play results.
    * A Tkinter-based GUI to play against the AI, watch AI vs. AI games, manage training, and visualize policies.
    * Advanced features like tree reuse, prioritized experience replay, SWA, and curriculum learning.

* **Using CodingAPI with Connect4-AlphaZero:**
    1.  Launch CodingAPI.
    2.  Check the "Upload existing program to modify" box.
    3.  Click "Browse..." and select the `Connect4/Connect4-AlphaZero.py` file.
    4.  In "Project name", you could enter `Connect4_Refinement`.
    5.  In "Program Description/Comment", you could specify a task, for example:
        `Refactor the MCTS search function in Connect4-AlphaZero.py to improve its performance by batching neural network evaluations for leaf nodes encountered during a single search pass.`
    6.  Select "Python" as the language.
    7.  Choose your desired Coding and Auditing LLMs and the number of iterations.
    8.  Click "Start Coding". CodingAPI will attempt to modify the existing `Connect4-AlphaZero.py` code according to your instructions, iterating through audit and correction cycles.

## DBrowser Application

* **Location:** The `DBrowser_2.0.py` script is located in the `DBrowser/` subdirectory.
* **Purpose:** DBrowser is a utility application designed to help you browse, explore, and manage the data generated by CodingAPI and stored in its SQLite database (`codingapi.db`).

* **Key Features of DBrowser:**
    * **Database Connection:** Allows browsing to and connecting with a `codingapi.db` file.
    * **Project Listing:** Displays a filterable and searchable list of all projects recorded by CodingAPI, showing project ID, name, mode (correction/creation), creation date, and status.
    * **Project Details View:**
        * Shows comprehensive information for a selected project, including its description, output directory, LLMs used, and start/end times.
    * **Iteration Exploration (Correction Mode):**
        * For projects run in "Multiple Correction" mode, it lists all iterations.
        * Displays the code generated and the audit report for each selected iteration.
        * Shows bug counts (critical, serious, non-critical, suggestions) and the number of fixed issues per category for each iteration.
    * **Model Results (Creation Mode):**
        * For projects run in "Multiple Creation" mode, it lists the code generated by each selected LLM and its status (completed/failed).
    * **Code and Audit Viewing:** A dedicated tab allows viewing the full code or audit text for the selected project, iteration, or model result.
    * **Run Application:** Includes a "Run App" button to directly execute Python code selected within the browser. This is useful for testing generated scripts.
    * **Data Export:** Functionality to export the viewed code or audit text to a file.
    * **Search and Filter:** Allows users to filter the project list by mode (correction/creation) and search by project name or description.

* **Running DBrowser:**
    ```bash
    python DBrowser/DBrowser_2.0.py
    ```
    Upon launching, DBrowser will attempt to automatically locate and connect to a `codingapi.db` file in common locations. If not found, you can use the "Browse..." button to select the database file (typically located in `CodingAPI/data/codingapi.db`).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (assuming a standard MIT License).
