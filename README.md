# Data analysis
- Document here the project: vocal_patterns
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Install

Go to `https://github.com/ElsaGregoire/vocal_patterns` to see the project, manage issues,

Clone the project and install it:

```bash
git clone https://github.com/ElsaGregoire/vocal_patterns.git
cd vocal_patterns            # install and test
```
# Setup

1. Create virtualenv and install the project:
```bash
pyenv virtualenv 3.10.6 vocal_patterns
pyenv activate vocal_patterns
pip install -r requirements.txt
```

2. place the raw audio folders in the data folder

3. run `make create_csv` to create the csv file with data paths and labels