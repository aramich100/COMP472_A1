
# COMP 472 - Assignment 1 (Winter 2021)

<img src="https://fas.concordia.ca/adfs/portal/logo/logo.png?id=728F70A3E333A7E7AB58C4185D855224308D7AA511313D14AFF478183F60D900">
 
 
# Team Members
- Michael Arabian
- Thomas Le
- Andre Saad


# Instructions

  For this assignment, we used Python along with the Scikit-learn machine learning framework to experiment with two different machine learning algorithms. We used a provided sentiment data set. The focus of this assignment was to gain experience on experimentations and analysis. See http://scikit-learn.org/stable/ for official documentation.
  
  <br /> <br />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width="400" height="200">  
  
   <br /> <br />
   
  # How to run our program
  
- Download our GitHub repository as a Zip or use 'Git Clone' to have a copy on your computer.
- Make sure to have Python 3.9.2 installed on your computer. If you do not have it, you can install it here : https://www.python.org/downloads/
- Run the following command in your terminal in order to install scikit-learn : 
  ```
pip install scikit-learn

```

 <br />  <br />
 # Our Results 
 
As instructed, using the SciKit Framework, we were able to run 3 different Machine Learning Algorithms and obtained very promising results.
 
##### Naive Bayes 
- Accuracy: 80.65463701216954
- Confusion Matrix: [ [ 1006  224 ][ 237  916 ]]
 
##### Decision Tree 
- Accuracy: 72.2198908938313
- Confusion Matrix: [ [ 870 360 ] [ 302 851 ]]

##### Better Decision Tree 
- Accuracy: 73.46454049517415
- Confusion Matrix: [ [ 868 362 ] [ 318 835 ]]

<br />
 We can see that the Naive Bayes algorithm held the highest accuracy while compared to the Decision Tree. This is due to the fact that the Decision Tree is a discriminative model, whereas the Naive Bayes is a generative model. Given our data set, the Naive Bayes is best suited for the highest accuracy. 
