# Question 6.4

# Simple prediction function 
# Input should be an array of length 2 with x[0]=longitude and x[1]=latitude
# Output is either a 0 (for blue) or a 1 (for red)
def predict(x):

    if x[1] > 36:
        # Latitude is above 36, check if longitude is above/below -97
        if x[0] > -97:
            # Longitude is above -97, classify as "blue", or 0
            y = 0
        else:
            # Longitude is below or equal to -97, classify as "red", or 1
            y = 1
    else:
        # Latitude is below or equal to 36, check if latitude is above/below 29
        if x[1] > 29:
            # Latitude is above 29, classify as "red", or 1
            y = 1
        else:
            # Latitude is below or equal to 29, classify as "blue", or 0
            y = 0
    
    return y

# Testing predict function
x = [-140,65]
print (str(x) + " -> " + str(predict(x)))
x = [-140,25]
print (str(x) + " -> " + str(predict(x)))
x = [-96,65]
print (str(x) + " -> " + str(predict(x)))
x = [-96,30]
print (str(x) + " -> " + str(predict(x)))
x = [-96,27]
print (str(x) + " -> " + str(predict(x)))
