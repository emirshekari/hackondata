from numpy.linalg import eigh

def estimateCovariance(data):
    """Compute the covariance matrix for a given rdd.

    Note:
        The multi-dimensional covariance array should be calculated using outer products.  Don't
        forget to normalize the data by first subtracting the mean.

    Args:
        data (RDD of np.ndarray):  An `RDD` consisting of NumPy arrays.

    Returns:
        np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
            length of the arrays in the input `RDD`.
    """
    # normalization or centering
    meanData = data.mean()
    dataZeroMean = data.map(lambda row : row - meanData)
    n = data.count()
    correlatedCov = dataZeroMean.map(lambda row: np.outer(row, row)).reduce(lambda mat1,mat2: mat1 + mat2)/float(n)
    return correlatedCov
def pca(data, k=2):
    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    Note:
        All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
        each eigenvectors as a column.  This function should also return eigenvectors as columns.

    Args:
        data (RDD of np.ndarray): An `RDD` consisting of NumPy arrays.
        k (int): The number of principal components to return.

    Returns:
        tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
            scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
            rows equals the length of the arrays in the input `RDD` and the number of columns equals
            `k`.  The `RDD` of scores has the same number of rows as `data` and consists of arrays
            of length `k`. Eigenvalues is an array of length d (the number of features).
    """
    covarianceMatrix = estimateCovariance(data)
    eigVals, eigVecs = eigh(covarianceMatrix)
    inds = np.argsort(-eigVals)
    # Return the `k` principal components, `k` scores, and all eigenvalues
    k_topComponents = eigVecs[:,inds[0:k]] # `k` principal components are the top `k` eigen vectors of the covaruance matrix
    correlatedDataScores = data.map(lambda row: row.dot(k_topComponents)) # scores are the dot product of the original data and the principle components
    descending_eigVals = eigVals[inds[::1]] # eigen values of the covariance matrix
    return k_topComponents, correlatedDataScores, descending_eigVals
def projectPointsAndGetLines(data, components, xRange):
    """Project original data onto first component and get line details for top two components."""
    topComponent= components[:, 0]
    slope1, slope2 = components[1, :2] / components[0, :2]

    means = data.mean()[:2]
    demeaned = data.map(lambda v: v - means)
    projected = demeaned.map(lambda v: (v.dot(topComponent) /
                                        topComponent.dot(topComponent)) * topComponent)
    remeaned = projected.map(lambda v: v + means)
    x1,x2 = zip(*remeaned.collect())

    lineStartP1X1, lineStartP1X2 = means - np.asarray([xRange, xRange * slope1])
    lineEndP1X1, lineEndP1X2 = means + np.asarray([xRange, xRange * slope1])
    lineStartP2X1, lineStartP2X2 = means - np.asarray([xRange, xRange * slope2])
    lineEndP2X1, lineEndP2X2 = means + np.asarray([xRange, xRange * slope2])

    return ((x1, x2), ([lineStartP1X1, lineEndP1X1], [lineStartP1X2, lineEndP1X2]),
            ([lineStartP2X1, lineEndP2X1], [lineStartP2X2, lineEndP2X2]))
    import matplotlib.pyplot as plt
import numpy as np

def create2DGaussian(mn, variance, cov, n):
    """Randomly sample points from a two-dimensional Gaussian distribution"""
    # mn is mean 
    # numpy.random.multivariate_normal(mean_vector, cov_matrix, size)
    #output is a matrix with n rows and 2 columns because cov_matrix is 2x2
    np.random.seed(142)
    return np.random.multivariate_normal(np.array([mn, mn]), np.array([[variance, cov], [cov, variance]]), n)

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax
# dataCorrelated is a 100x2 matrix
dataCorrelated = create2DGaussian(mn=50, variance=1, cov=.9, n=100)
correlatedData = sc.parallelize(dataCorrelated)

topComponentsCorrelated, correlatedDataScoresAuto, eigenvaluesCorrelated = pca(correlatedData, 2)

((x1, x2), (line1X1, line1X2), (line2X1, line2X2)) = \
    projectPointsAndGetLines(correlatedData, topComponentsCorrelated, 5)

# generate layout and plot data
fig, ax = preparePlot(np.arange(46, 55, 2), np.arange(46, 55, 2), figsize=(7, 7))
ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
plt.plot(line1X1, line1X2, linewidth=3.0, c='#8cbfd0', linestyle='--')
plt.plot(line2X1, line2X2, linewidth=3.0, c='#d6ebf2', linestyle='--')
plt.scatter(dataCorrelated[:,0], dataCorrelated[:,1], s=14**2, c='#d6ebf2',
            edgecolors='#8cbfd0', alpha=0.75)
plt.scatter(x1, x2, s=14**2, c='#62c162', alpha=.75)
display(fig) 
pass

import os
baseDir = os.path.join('databricks-datasets', 'cs190')
inputPath = os.path.join('data-001', 'neuro.txt')
inputFile = os.path.join(baseDir, inputPath)
lines = sc.textFile(inputFile)

def parse(line):
    """Parse the raw data into a (`tuple`, `np.ndarray`) pair.

    Note:
        You should store the pixel coordinates as a tuple of two ints and the elements of the pixel intensity
        time series as an np.ndarray of floats.

    Args:
        line (str): A string representing an observation. Elements are separated by spaces.  The
            first two elements represent the coordinates of the pixel, and the rest of the elements
            represent the pixel intensity over time.

    Returns:
        tuple of tuple, np.ndarray: A (coordinate, pixel intensity array) `tuple` where coordinate is
            a `tuple` containing two values and the pixel intensity is stored in an NumPy array
            which contains 240 values.
    """
    list_of_data = line.split() # makes a list 
    coordinate = tuple([int(x) for x in list_of_data[0:2]])
    pixel_intensity = [float(x) for x in list_of_data[2:]]
    pixel_intensity = np.asarray(pixel_intensity)# convert list to np array
    return (coordinate, pixel_intensity)

rawData = lines.map(parse)
rawData.cache()
entry = rawData.first()
print ('Length of movie is {0} seconds'.format(len(entry[1])))
print ('Number of pixels in movie is {0:,}'.format(rawData.count()))
print ('\nFirst entry of rawData (with only the first five values of the NumPy array):\n({0}, {1})'
       .format(entry[0], entry[1][:5]))
mn = rawData.map(lambda coor_timeseries: np.amin(coor_timeseries[1])).reduce(lambda min1,min2: min(min1,min2)) #"reduce" finds the min among 46460 mins of each record
mx = rawData.map(lambda coor_timeseries: np.amax(coor_timeseries[1])).reduce(lambda max1,max2: max(max1,max2))
print (mn, mx)

example = rawData.filter(lambda coor_timeseries_tuple: np.std(coor_timeseries_tuple[1]) > 100).values().first()

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 300, 50), np.arange(300, 800, 100))
ax.set_xlabel(r'time'), ax.set_ylabel(r'fluorescence')
ax.set_xlim(-20, 270), ax.set_ylim(270, 730)
plt.plot(range(len(example)), example, c='#8cbfd0', linewidth='3.0')
display(fig) 
pass

def rescale(ts):
    """Take a np.ndarray and return the standardized array by subtracting and dividing by the mean.

    Note:
        You should first subtract the mean and then divide by the mean.

    Args:
        ts (np.ndarray): Time series data (`np.float`) representing pixel intensity.

    Returns:
        np.ndarray: The times series adjusted by subtracting the mean and dividing by the mean.
    """
    average = np.mean(ts)
    return (ts - average)/float(average)

scaledData = rawData.mapValues(lambda v: rescale(v))
mnScaled = scaledData.map(lambda coor_ts_tuple: coor_ts_tuple[1]).map(lambda v: min(v)).min()
mxScaled = scaledData.map(lambda coor_ts_tuple: coor_ts_tuple[1]).map(lambda v: max(v)).max()
print (mnScaled, mxScaled)

example = scaledData.filter(lambda coor_ts_tuple: np.std(coor_ts_tuple[1]) > 0.1).values().first()

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 300, 50), np.arange(-.1, .6, .1))
ax.set_xlabel(r'time'), ax.set_ylabel(r'fluorescence')
ax.set_xlim(-20, 260), ax.set_ylim(-.12, .52)
plt.plot(range(len(example)), example, c='#8cbfd0', linewidth='3.0')
display(fig) 
pass

# Run pca using scaledData
scaledData_values = scaledData.map(lambda key_value_tuple: key_value_tuple[1])
componentsScaled, scaledScores, eigenvaluesScaled = pca(scaledData_values, 3) 
# componentsScaled is eigenvectors,    scaledScores is "representation of original data in the compact space using pca"

import matplotlib.cm as cm

scoresScaled = np.vstack(scaledScores.collect()) # vstack creates a matrix
print(scoresScaled.shape)

imageOneScaled = scoresScaled[:,0].reshape(230, 202).T # scoresScaled[:,0]: every row but only in column 0
# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
ax.set_title('Top Principal Component', color='#888888') 
image = plt.imshow(imageOneScaled,interpolation='nearest', aspect='auto', cmap=cm.gray)
display(fig) 
pass

imageTwoScaled = scoresScaled[:,1].reshape(230, 202).T

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
ax.set_title('Second Principal Component', color='#888888')
image = plt.imshow(imageTwoScaled,interpolation='nearest', aspect='auto', cmap=cm.gray)
display(fig) 
pass

# Adapted from python-thunder's Colorize.transform where cmap='polar'.
# Checkout the library at: https://github.com/thunder-project/thunder and
# http://thunder-project.org/

def polarTransform(scale, img):
    """Convert points from cartesian to polar coordinates and map to colors."""
    from matplotlib.colors import hsv_to_rgb

    img = np.asarray(img)
    dims = img.shape

    phi = ((np.arctan2(-img[0], -img[1]) + np.pi/2) % (np.pi*2)) / (2 * np.pi)
    rho = np.sqrt(img[0]**2 + img[1]**2)
    saturation = np.ones((dims[1], dims[2]))

    out = hsv_to_rgb(np.dstack((phi, saturation, scale * rho)))

    return np.clip(out * scale, 0, 1)
    
    # Show the polar mapping from principal component coordinates to colors.
x1AbsMax = np.max(np.abs(imageOneScaled)) # = 4.014
x2AbsMax = np.max(np.abs(imageTwoScaled)) # = 2.382

numOfPixels = 300
x1Vals = np.arange(-x1AbsMax, x1AbsMax, (2 * x1AbsMax) / numOfPixels)
x2Vals = np.arange(x2AbsMax, -x2AbsMax, -(2 * x2AbsMax) / numOfPixels)
x2Vals.shape = (numOfPixels, 1)

x1Data = np.tile(x1Vals, (numOfPixels, 1))
x2Data = np.tile(x2Vals, (1, numOfPixels))

# Try changing the first parameter to lower values
polarMap = polarTransform(1.0, [x1Data, x2Data])

gridRange = np.arange(0, numOfPixels + 25, 25)
fig, ax = preparePlot(gridRange, gridRange, figsize=(9.0, 7.2), hideLabels=True)
image = plt.imshow(polarMap, interpolation='nearest', aspect='auto')
ax.set_xlabel('Principal component one'), ax.set_ylabel('Principal component two')
gridMarks = (2 * gridRange / float(numOfPixels) - 1.0)
x1Marks = x1AbsMax * gridMarks
x2Marks = -x2AbsMax * gridMarks
ax.get_xaxis().set_ticklabels(map(lambda x: '{0:.1f}'.format(x), x1Marks))
ax.get_yaxis().set_ticklabels(map(lambda x: '{0:.1f}'.format(x), x2Marks))
ax.set_title('How low-dimensional representations, which correspond to \n the first two principal components, are mapped to colors \n', color='#465150') 

display(fig)  
pass

# Use the same transformation on the image data
# Try changing the first parameter to lower values
brainmap = polarTransform(1.0, [imageOneScaled, imageTwoScaled])

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
image = plt.imshow(brainmap,interpolation='nearest', aspect='auto')
display(fig) 
pass

# Create a multi-dimensional array to perform the aggregation using these:
# np.eye
# np.tile
# np.kron
T = np.tile(np.eye(20),12)

# Transform scaledData using T.  Make sure to retain the keys.
timeData = scaledData.map(lambda coordinate_intensity_tuple: (coordinate_intensity_tuple[0], T.dot(coordinate_intensity_tuple[1])))

timeData.cache()
print (timeData.count())
print (timeData.first())

timeData_intensity = timeData.map(lambda coordinate_intensity_tuple: coordinate_intensity_tuple[1])
componentsTime, timeScores, eigenvaluesTime = pca(timeData_intensity, 3)

print ('componentsTime: (first five) \n{0}'.format(componentsTime[:5,:]))
print ('\ntimeScores (first three): \n{0}'
       .format('\n'.join(map(str, timeScores.take(3)))))
print ('\neigenvaluesTime: (first five) \n{0}'.format(eigenvaluesTime[:5]))

scoresTime = np.vstack(timeScores.collect())
imageOneTime = scoresTime[:,0].reshape(230, 202).T
imageTwoTime = scoresTime[:,1].reshape(230, 202).T
brainmap = polarTransform(3, [imageOneTime, imageTwoTime])

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
image = plt.imshow(brainmap,interpolation='nearest', aspect='auto')
display(fig) 
pass

# Create a multi-dimensional array to perform the aggregation
D = np.kron(np.eye(12),np.ones(20))

# Transform scaledData using D.  Make sure to retain the keys.
directionData = scaledData.map(lambda coordinate_intensity_tuple: (coordinate_intensity_tuple[0], D.dot(coordinate_intensity_tuple[1])))

directionData.cache()
print (directionData.count())
print (directionData.first())

directionData_intensity = directionData.map(lambda coordinate_intensity_tuple: coordinate_intensity_tuple[1])
componentsDirection, directionScores, eigenvaluesDirection = pca(directionData_intensity , 3)

print ('componentsDirection: (first five) \n{0}'.format(componentsDirection[:5,:]))
print ('\ndirectionScores (first three): \n{0}'
       .format('\n'.join(map(str, directionScores.take(3)))))
print ('\neigenvaluesDirection: (first five) \n{0}'.format(eigenvaluesDirection[:5]))

scoresDirection = np.vstack(directionScores.collect())
imageOneDirection = scoresDirection[:,0].reshape(230, 202).T
imageTwoDirection = scoresDirection[:,1].reshape(230, 202).T
brainmap = polarTransform(2, [imageOneDirection, imageTwoDirection])
# with thunder: Colorize(cmap='polar', scale=2).transform([imageOneDirection, imageTwoDirection])

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
image = plt.imshow(brainmap, interpolation='nearest', aspect='auto')
display(fig) 
pass

