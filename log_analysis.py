# perform web server log analysis with Apache Spark (in Python)
# In this code, I use Apache Spark on real-world text-based production logs and fully harness the power of that data.
import re
import datetime
from pyspark.sql import Row
month_map = {'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
    'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12}

def parse_apache_time(s):
    """ Convert Apache time format into a Python datetime object
    Args:
        s (str): date and time in Apache time format
    Returns:
        datetime: datetime object (ignore timezone for now)
    """
    return datetime.datetime(int(s[7:11]),
                             month_map[s[3:6]],
                             int(s[0:2]),
                             int(s[12:14]),
                             int(s[15:17]),
                             int(s[18:20]))

def parseApacheLogLine(logline):
    """ Parse a line in the Apache Common Log format
    Args:
        logline (str): a line of text in the Apache Common Log format
    Returns:
        tuple: either a dictionary containing the parts of the Apache Access Log and 1,
               or the original invalid log line and 0
    """
    match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)
    if match is None:
        return (logline, 0)
    size_field = match.group(9)
    if size_field == '-':
        size = long(0)
    else:
        size = long(match.group(9))
    return (Row(
        host          = match.group(1),
        client_identd = match.group(2),
        user_id       = match.group(3),
        date_time     = parse_apache_time(match.group(4)),
        method        = match.group(5),
        endpoint      = match.group(6),
        protocol      = match.group(7),
        response_code = int(match.group(8)),
        content_size  = size
    ), 1)

# A regular expression pattern to extract fields from the log line
APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)\s*" (\d{3}) (\S+)'

#(1b) Configuration and Initial RDD Creation
import sys
import os
from test_helper import Test
baseDir = os.path.join('databricks-datasets')
inputPath = os.path.join('cs100', 'lab2', 'data-001', 'apache.access.log.PROJECT')
logFile = os.path.join(baseDir, inputPath)
def parseLogs():
    """ Read and parse log file """
    parsed_logs = (sc
                   .textFile(logFile)
                   .map(parseApacheLogLine)
                   .cache())
    access_logs = (parsed_logs
                   .filter(lambda s: s[1] == 1)
                   .map(lambda s: s[0])
                   .cache())
    failed_logs = (parsed_logs
                   .filter(lambda s: s[1] == 0)
                   .map(lambda s: s[0]))
    failed_logs_count = failed_logs.count()
    if failed_logs_count > 0:
        print 'Number of invalid logline: %d' % failed_logs.count()
        for line in failed_logs.take(20):
            print 'Invalid logline: %s' % line
    print 'Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (parsed_logs.count(), access_logs.count(), failed_logs.count())
    return parsed_logs, access_logs, failed_logs
parsed_logs, access_logs, failed_logs = parseLogs()

#(1c) Data Cleaning
APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)\s*" (\d{3}) (\S+)'
parsed_logs, access_logs, failed_logs = parseLogs()

#Part 2: Sample Analyses on the Web Server Log File
# Calculate statistics based on the content size.
content_sizes = access_logs.map(lambda log: log.content_size).cache()
print 'Content Size Avg: %i, Min: %i, Max: %s' % (
    content_sizes.reduce(lambda a, b : a + b) / content_sizes.count(),
    content_sizes.min(),
    content_sizes.max())

#(2b) Example: Response Code Analysis
# Response Code to Count
# a pair RDD is generated 
responseCodeToCount = (access_logs
                       .map(lambda log: (log.response_code, 1))
                       .reduceByKey(lambda a, b : a + b)
                       .cache())
responseCodeToCountList = responseCodeToCount.take(100)
print 'Found %d response codes' % len(responseCodeToCountList)
print 'Response Code Counts: %s' % responseCodeToCountList
assert len(responseCodeToCountList) == 7
assert sorted(responseCodeToCountList) == [(200, 940847), (302, 16244), (304, 79824), (403, 58), (404, 6185), (500, 2), (501, 17)]

#(2c) Example: Response Code Graphing with matplotlib
labels = responseCodeToCount.map(lambda (x, y): x).collect()
print labels
count = access_logs.count()
fracs = responseCodeToCount.map(lambda (x, y): (float(y) / count)).collect()
print fracs

import matplotlib.pyplot as plt
def pie_pct_format(value):
    """ Determine the appropriate format string for the pie chart percentage label
    Args:
        value: value of the pie slice
    Returns:
        str: formated string label; if the slice is too small to fit, returns an empty string for label
    """
    return '' if value < 7 else '%.0f%%' % value
fig = plt.figure(figsize=(4.5, 4.5), facecolor='white', edgecolor='white')
colors = ['yellowgreen', 'lightskyblue', 'gold', 'purple', 'lightcoral', 'yellow', 'black']
explode = (0.05, 0.05, 0.1, 0, 0, 0, 0)
patches, texts, autotexts = plt.pie(fracs, labels=labels, colors=colors,
                                    explode=explode, autopct=pie_pct_format,
                                    shadow=False,  startangle=125)
for text, autotext in zip(texts, autotexts):
    if autotext.get_text() == '':
        text.set_text('')  # If the slice is small to fit, don't show a text label
plt.legend(labels, loc=(0.80, -0.1), shadow=True)
display(fig)  
pass
# Create a DataFrame and visualize using display()
responseCodeToCountRow = responseCodeToCount.map(lambda (x, y): Row(response_code=x, count=y))
responseCodeToCountDF = sqlContext.createDataFrame(responseCodeToCountRow)
display(responseCodeToCountDF)

#(2d) Example: Frequent Hosts 
# Any hosts that has accessed the server more than 10 times.
hostCountPairTuple = access_logs.map(lambda log: (log.host, 1))
hostSum = hostCountPairTuple.reduceByKey(lambda a, b : a + b)
hostMoreThan10 = hostSum.filter(lambda s: s[1] > 10)
hostsPick20 = (hostMoreThan10
               .map(lambda s: s[0])
               .take(20))
print 'Any 20 hosts that have accessed more then 10 times: %s' % hostsPick20

#(2e) Example: Visualizing Endpoints
endpoints = (access_logs
             .map(lambda log: (log.endpoint, 1))
             .reduceByKey(lambda a, b : a + b)
             .cache())
ends = endpoints.map(lambda (x, y): x).collect()
counts = endpoints.map(lambda (x, y): y).collect()
#print ends[0:5]
fig = plt.figure(figsize=(8,4.2), facecolor='white', edgecolor='white')
# len(ends) : number of members of the list ends
plt.axis([0, len(ends), 0, max(counts)])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Endpoints')
plt.ylabel('Number of Hits')
plt.plot(counts)
display(fig)  
pass

#**(2e-dbc) Visualizing Endpoints using Databricks Cloud Plots**
# Create an RDD with Row objects
endpoint_counts_rdd = endpoints.map(lambda s: Row(endpoint = s[0], num_hits = s[1]))
endpoint_counts_schema_rdd = sqlContext.createDataFrame(endpoint_counts_rdd)

# Display a plot of the distribution of the number of hits across the endpoints.
display(endpoint_counts_schema_rdd)

#(2f) Top Endpoints
endpointCounts = (access_logs
                  .map(lambda log: (log.endpoint, 1))
                  .reduceByKey(lambda a, b : a + b))
topEndpoints = endpointCounts.takeOrdered(10, lambda s: -1 * s[1])
print 'Top Ten Endpoints: %s' % topEndpoints
assert topEndpoints == [(u'/images/NASA-logosmall.gif', 59737), (u'/images/KSC-logosmall.gif', 50452), (u'/images/MOSAIC-logosmall.gif', 43890), (u'/images/USA-logosmall.gif', 43664), (u'/images/WORLD-logosmall.gif', 43277), (u'/images/ksclogo-medium.gif', 41336), (u'/ksc.html', 28582), (u'/history/apollo/images/apollo-logo1.gif', 26778), (u'/images/launch-logo.gif', 24755), (u'/', 20292)], 'incorrect Top Ten Endpoints'

##### **Part 3: Analyzing Web Server Log File**
#(3a) Exercise: Top Ten Error Endpoints
not200 = (access_logs
          .map(lambda log: (log.response_code,log.endpoint))
          .filter(lambda (x,y): x != 200))
endpointCountPairTuple = not200.map(lambda (x,y):(y,1))
endpointSum = endpointCountPairTuple.reduceByKey(lambda x,y: x+y)
topTenErrURLs = endpointSum.takeOrdered(10,key = lambda x: -x[1])
print 'Top Ten failed URLs: %s' % topTenErrURLs

#(3b) Exercise: Number of Unique Hosts
hosts = (access_logs
        .map(lambda log: (log.host,1))
        .reduceByKey(lambda x,y : x+y))
#uniqueHosts = hosts.map(lambda (x,y): x)
uniqueHostCount = hosts.count()
print 'Unique hosts: %d' % uniqueHostCount

#(3c) Number of Unique Daily Hosts
dayToHostPairTuple=access_logs.map(lambda log:(log.date_time.day,log.host))
#below Group the values for each key in the RDD into a single sequence.	
dayGroupedHosts=dayToHostPairTuple.groupByKey()
#below v is a list and set(v) is another list with no repetition  TODO: Replace <FILL IN> with appropriate code
dayHostCount = dayGroupedHosts.map(lambda (k, v): (k, set(v)))
dayHostCount = dayHostCount.map(lambda (x,v):(x,len(v)))
#dailyHostsList = dailyHosts.take(30)
dailyHosts = (dayHostCount.sortByKey().cache())
dailyHostsList = dailyHosts.take(30)
print 'Unique hosts per day: %s' % dailyHostsList

#(3d) Visualizing the Number of Unique Daily Hosts
daysWithHosts = dailyHosts.map(lambda (k,v):k).collect()
hosts = dailyHosts.map(lambda (k,v):v).collect()
fig = plt.figure(figsize=(8,4.5), facecolor='white', edgecolor='white')
plt.axis([min(daysWithHosts), max(daysWithHosts), 0, max(hosts)+500])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Day')
plt.ylabel('Hosts')
plt.plot(daysWithHosts, hosts)
display(fig)  
pass

#(3e) Average Number of Daily Requests per Hosts 
dayAndHostTuple = access_logs.map(lambda log:(log.date_time.day,log.host))
groupedByDay = dayAndHostTuple.groupByKey()
sortedByDay = groupedByDay.sortByKey()
avgDailyReqPerHost=(sortedByDay.map(lambda (x,v):(x,len(v)))
                    .join(dailyHosts)
                    .sortByKey()
                    .map(lambda (k,(v1,v2)):(k,(v1/v2)))).cache()
avgDailyReqPerHostList = avgDailyReqPerHost.take(30)
print 'Average number of daily requests per Hosts is %s' % avgDailyReqPerHostList

#(3f) Visualizing the Average Daily Requests per Unique Host
daysWithAvg = avgDailyReqPerHost.map(lambda (x,v):x).collect()
avgs = avgDailyReqPerHost.map(lambda (x,v):v).collect()
fig = plt.figure(figsize=(8,4.2), facecolor='white', edgecolor='white')
plt.axis([0, max(daysWithAvg), 0, max(avgs)+2])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Day')
plt.ylabel('Average')
plt.plot(daysWithAvg, avgs)
display(fig)  
pass

# Part 4: Exploring 404 Response Codes
# 4a: Counting 404 Response Codes
badRecords = (access_logs
              .filter(lambda log:log.response_code==404)).cache()
print 'Found %d 404 URLs' % badRecords.count()
#(4b) Listing 404 Response Code Records 
badEndpoints = badRecords.map(lambda log:(log.endpoint,1)).reduceByKey(lambda a,b:a+b)
badUniqueEndpoints = badEndpoints.map(lambda (a,b):a)
badUniqueEndpointsPick40 = badUniqueEndpoints.take(40)
print '404 URLS: %s' % badUniqueEndpointsPick40

#(4c) Listing the Top Twenty 404 Response Code Endpoints
badEndpointsCountPairTuple = badRecords.map(lambda log:(log.endpoint,1))
badEndpointsSum = badEndpointsCountPairTuple.reduceByKey(lambda a,b:a+b)
badEndpointsTop20 = badEndpointsSum.takeOrdered(20, lambda s:-1*s[1])
print 'Top Twenty 404 URLs: %s' % badEndpointsTop20

#(4d) Listing the Top Twenty-five 404 Response Code Hosts
errHostsCountPairTuple = badRecords.map(lambda x:(x.host,1))
errHostsSum = errHostsCountPairTuple.reduceByKey(lambda a,b:a+b)
errHostsTop25 = errHostsSum.takeOrdered(25, lambda s:-1*s[1])
print 'Top 25 hosts that generated errors: %s' % errHostsTop25

#(4e) Listing 404 Response Codes per Day 
errDateCountPairTuple = badRecords.map(lambda log:(log.date_time.day,log.host)).groupByKey()
errDateSum = errDateCountPairTuple.map(lambda (x,v):(x,len(v)))
errDateSorted = (errDateSum.sortByKey().cache())
errByDate = errDateSorted.collect()
print '404 Errors by day: %s' % errByDate

#(4f) Visualizing the 404 Response Codes by Day 
daysWithErrors404 = errDateSorted.map(lambda (x,v):x).collect()
errors404ByDay = errDateSorted.map(lambda (x,v):v).collect()
fig = plt.figure(figsize=(8,4.2), facecolor='white', edgecolor='white')
plt.axis([0, max(daysWithErrors404), 0, max(errors404ByDay)])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Day')
plt.ylabel('404 Errors')
plt.plot(daysWithErrors404, errors404ByDay)
display(fig)  
pass

#(4g) Top Five Days for 404 Response Codes
topErrDate = errDateSorted.takeOrdered(5, lambda s:-1*s[1])
print 'Top Five dates for 404 requests: %s' % topErrDate

#(4h) Hourly 404 Response Codes
hourCountPairTuple = badRecords.map(lambda log:(log.date_time.hour,log.response_code))
hourRecordsSum = hourCountPairTuple.groupByKey().map(lambda (x,v):(x,len(v)))
hourRecordsSorted = (hourRecordsSum.sortByKey().cache())
errHourList = hourRecordsSorted.collect()
print 'Top hours for 404 requests: %s' % errHourList

#(4i) Visualizing the 404 Response Codes by Hour
hoursWithErrors404 = hourRecordsSorted.map(lambda (x,v):x).collect()
errors404ByHours = hourRecordsSorted.map(lambda (x,v):v).collect()
fig = plt.figure(figsize=(8,4.2), facecolor='white', edgecolor='white')
plt.axis([0, max(hoursWithErrors404), 0, max(errors404ByHours)])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Hour')
plt.ylabel('404 Errors')
plt.plot(hoursWithErrors404, errors404ByHours)
display(fig)  
pass
