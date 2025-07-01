# Install Ioniq

This guide walks you through the steps to download, set up, and install the Ioniq package using a virtual environment.

---

## 1. Download the Ioniq Repository

- Open the WanunuLab GitHub repository:  
  https://github.com/wanunulab/ioniq

- Click the green **Code** button and choose **Download ZIP**.

![](../images/download_repo.png)

- Unzip the folder and move it to your desired location (e.g., Desktop or Documents).

---

## 2. Open the Ioniq Folder in VS Code

- Launch the **VS Code** application.
- Click **File > Open...** and select the unzipped `ioniq` folder.

![](../../../Desktop/Screenshot 2025-07-01 at 11.24.59 AM.png)

Now your working directory in VS Code is set to the `ioniq` folder.

---

## 3. Set Up a Virtual Environment

A **virtual environment** is an isolated workspace for installing project-specific dependencies.

> Learn more: [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

### Step 1: Open Terminal in VS Code

- Go to the **top menu** and click **Terminal > New Terminal**.

![](../images/open_terminal.png)

### Step 2: Confirm You're in the Correct Directory

Use the following command:

```bash
ls
```

You should see the contents of the `ioniq` folder.

![](../images/list_dir.png)
### Step 3: Create and Activate the Environment

Run the command:

```bash
python -m venv ioniq_env
```
> **Note:** On some systems (especially Linux/macOS), you may need to use `python3` and `pip3` instead of `python` and `pip`.

Then activate it:

- **Linux / macOS:**
  ```bash
  source ioniq_env/bin/activate
  ```

- **Windows:**
  ```bat
  ioniq_env\Scripts\activate
  ```

If activated successfully, you’ll see `(ioniq_env)` appear before your username in the terminal.
![](../images/active_env.png)


---

## 4. Install the Ioniq Package

Run the command:

```bash
pip install .
```

Wait a few minutes while dependencies install.

![](../images/final_ioniq_installes.png)

> If you see a similar message, congratulations — you've successfully installed the Ioniq package!

---

## 5. (Optional) Install JupyterLab

To run data analysis in an interactive interface like Jupyter Notebook, install JupyterLab while the virtual environment is still active:

```bash
pip install jupyterlab
```

---

## 6. Deactivate the Environment

When you're done, deactivate the virtual environment by typing:

```bash
deactivate
```

> It's good practice to **activate** your virtual environment when working on a project and **deactivate** it when you're finished.
