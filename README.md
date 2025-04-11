# TBoD-Lab-5-Exploring-Machine-Learning

(1) It's a penguins Dataset
(2) It is pip installed,
(3) Problem, I want to solve -- recognizing if it's belongs to the group of Adelie, Gentoo or Chinstrap

# Project-A---Visualizing-Air-Pollution-Data

***Names: Nice Teta Hirwa*** <br />
***Instructor: Professor Mike Ryu*** <br />
***Class: CS-150*** <br />


## Thesis Statement
This is a machine learning dashboard that depicts the prediction of species such as Adelie, Chinstrap, and Gentoo 
basing on their flipper length and bill depth (mm). 

## Context of my data visualization
This dashboard employs svm classification algorithm by using the palmer penguins dataset that is inherently built into pycharm. 
After loading the dataset, I trained it where x was the columns that characterizes the penguins such bill depth and flipper length,
and y or the target was the species. After training, I tested it out and the dashboard displays the results of species categorization. 

## Data I will be visualizing
The data displays how well the svm categorizes the species depending on the flipper length and bill depth(mm). The color coded 
regions (red, blue, gray) represent the classification areas created by the SVM model. The boundary lines represent a threshold that 
visually demonstrates how the model separates different classes based on the input features: Bill length (mm) on the x-axis 
and flipper length (mm) on the y-axis.

## Explaining the coding part of the project
app.py has three main parts that makes the visual display effective. 
1. Load_data: I imported the palmer penguins since it is built into pycharm and removed all the rows that had missing data.  
2. app_layout: I designed the layout the same as the one in the book of dash chapter 7, added title, sliders, dropdowns, all on different cards. 
3. Call back: In my call back function, I have 8 call backs, about to 6 callbacks updates or disables the sliders for coefs, gamma, and resets the threshold center. The seventh one is for the default bottom, 
that defaults the back the data to the parameters that has the highest accuracy. And the last one updates the svm graph. This is where all the games happening from. It trains the dataset, tests the dataset and 
imports a file from the utils that computes the confusion matrix. 


## Source of the Data
1. Palmer Penguins dataset was built into pycharm!


And I finally called the main() function to run the code.

