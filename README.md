# Exploratory Analysis on Food.com Large Dataset

This repository contains source code for a simple yet visually appealing Python dashboard (built with [Flask](https://flask.palletsprojects.com/en/stable/)) visualizing the results of a unique exploratory analysis for a large dataset containing hundreds of thousands of food recipes from [Food.com](https://www.food.com/) downloaded through [Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews).

The dashboard is structured as PowerPoint-like with a total of five slides, including a home page, and can be navigated supposedly easy by providing a side navigation bar to jump to any slides or to come back to home. **Each slide contains (multiple) interactive and beautifully-designed figures** for the user to appreciate at. The figures are also accompanied by some detailed texts explaining the 5W1H of the corresponding analysis.

![Network Graph of Ingredients](/figures/network_graph.png)

## Running

The dashboard can be run with `Python 3.11` and it is recommended to do so within a virtual environment to avoid dependencies' conflicts. Install Flask first by running
```
pip install Flask
```
with a capital "F". It is enough to run the dashboard. However, if the user wants to run the full source code, install from `requirements.txt` instead.

The dashboard can then simply be run, in the parent folder, by typing
```
python app.py
```
The [plots](plots) folder already contains all the necessary JSON files to render the figures. 

The dataset itself is not uploaded to this repo due to size and copyright, hence if the user wants to download the dataset themselves, they can do so by navigating to [here](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) to download or they can simply run 
```
python download_data.py
```
and it will save the raw data to [/data/raw](/data/raw) folder. Then, run 
```
python preprocessing.py
```
to pre-process the raw data and store it as an SQLite database in [/data/tables](/data/tables) folder, ready to be processed further and analyzed by the other files.

## Content

The source codes for the analysis and figures generation are uploaded within the [src](src) folder. It contains four main `.py` files, each served as the source code for each panel slide in the dashboard (except homepage):
- [plot_ingredients.py](/src/plot_ingredients.py) for generating all figures within the ingredient network graph panel,
- [plot_nutrition.py](/src/plot_nutrition.py) for generating all figures within the nutritional landscape panel, including the PCA analysis,
- [plot_duration.py](/src/plot_duration.py) for generating all figures within the recipe's cooking time distribution panel, and
- [plot_features.py](/src/plot_features.py) for generating all figures within the feature importance panel, including both the sentiment and regression analysis.

Each of these files output a single JSON file, stored in [plots](plots), containing all the necessary data and metadata to render all the relevant figures with Plotly in the dashboard, avoiding the need to do the heavy work on the front-end. A file named `content.py` contains almost all the texts shown in the dashboard, including the explanatory text for each figure. Note that the texts are suitable for the specific analysis setup and figures' configuration that originally used and uploaded to this repo, hence changing or fine-tuning any parameters or hyperparameters can make the content useless.

[/static/css](/static/css) contains `style.css`, the corresponding CSS file for the dashboard styling, while [templates](templates) contains `index.html`, a file with both the HTML and JavaScript combined to build the dashboard.

## Author's Remarks

As the homepage states, the analysis conducted to build this project is limited to be exploratory, not a real research topic whatsoever, and intially served solely as the author's learning experience, both in terms of data visualization and coding skill. 

If there are some problems or improvements to the analysis, I am widely opened to any suggestions and critics. Please feel free to reach me through salirafi8@gmail.com or just open an issue in this repo :)

The use of generative AI includes: Visual Studio Code's Copilot to help tidying up code and writing comments and docstring, as well as OpenAI's Chat GPT to help with code syntax ideas and identify runtime error. Outside of those, including problem formulation and framework of thinking, code logical reasoning and writing, from database management using SQLite to web development using Flask, all is done mostly by the author.

## Data Sources

Only one dataset used in this analysis: [The Food.com - Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) dataset available publicly through Kaggle.
