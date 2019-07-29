# Python 2 has two integer types int and long. These have been unified in Python 3, so there is now only one type, int.
# If you do require it the following code works on both Python 2 and Python 3 without 2to3 conversion:
import sys
if sys.version_info > (3,):
  long = int
  
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
        datetime: datetime object (ignore timezone for now)"""
    return datetime.datetime(int(s[7:11]), month_map[s[3:6]], int(s[0:2]), int(s[12:14]), int(s[15:17]), int(s[18:20]))
def parseApacheLogLine(logline):
    """ Parse a line in the Apache Common Log format
    Args:
        logline (str): a line of text in the Apache Common Log format
    Returns:
        tuple: either a dictionary containing the parts of the Apache Access Log and 1,
               or the original invalid log line and 0"""
    # #example log line:  `127.0.0.1 - - [01/Aug/1995:00:00:01 -0400] "GET /images/launch-logo.gif HTTP/1.0" 200 1839`
    match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)
    if match is None:
        return (logline, 0)
    size_field = match.group(9) # 1839 (it is the last field in the example log line)
    if size_field == '-':
        size = long(0)
    else:
        size = long(match.group(9)) 
    return (Row(
        host          = match.group(1), # 127.0.0.1
        client_identd = match.group(2), # -
        user_id       = match.group(3), # -
        date_time     = parse_apache_time(match.group(4)), # [01/Aug/1995:00:00:01 -0400]
        method        = match.group(5), # GET
        endpoint      = match.group(6), # /images/launch-logo.gif
        protocol      = match.group(7), # HTTP/1.0
        response_code = int(match.group(8)), # 200
        content_size  = size
    ), 1)
    
# A regular expression pattern to extract fields from the log line
APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)\s*" (\d{3}) (\S+)'

import os
#from test_helper import Test

baseDir = os.path.join('databricks-datasets')
inputPath = os.path.join('cs100', 'lab2', 'data-001', 'apache.access.log.PROJECT')
logFile = os.path.join(baseDir, inputPath)

def parseLogs():
    """ Read and parse log file """
    parsed_logs = (sc
                   .textFile(logFile)
                   .map(parseApacheLogLine)
                   .cache())       # parsed_logs is a tuple in the form of :
                                   #(Row(host, client_identd, user_id, date_time, method, endpoint, protocol, response_code,content_size), 1)

    access_logs = (parsed_logs
                   .filter(lambda s: s[1] == 1)
                   .map(lambda s: s[0])
                   .cache()) #a Row obcet in form of: Row(host, client_identd, user_id, date_time, method, endpoint, protocol, response_code,content_size)

    failed_logs = (parsed_logs 
                   .filter(lambda s: s[1] == 0)
                   .map(lambda s: s[0])) # failed_logs is in the form of a string (which is the log line) 
    failed_logs_count = failed_logs.count()
    if failed_logs_count > 0:
        print ('Number of invalid logline: %d' % failed_logs.count())
        for line in failed_logs.take(20):
            print ('Invalid logline: %s' % line)

    print ('Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (parsed_logs.count(), access_logs.count(), failed_logs.count()))
    return parsed_logs, access_logs, failed_logs

parsed_logs, access_logs, failed_logs = parseLogs()

logline ='ix-sac6-20.ix.netcom.com - - [08/Aug/1995:14:43:39 -0400] "GET / HTTP/1.0 " 200 7131'
APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)\s*" (\d{3}) (\S+)'
match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)
if match is None:
     print ('not matched')
else:
     print ('matched')
 
# Calculate statistics based on the content size.
content_sizes = access_logs.map(lambda log: log.content_size).cache()
print ('Content Size Avg: %i, Min: %i, Max: %s' % (
    content_sizes.reduce(lambda a, b : a + b) / content_sizes.count(),
    content_sizes.min(),
    content_sizes.max()))
    
# Response Code to Count
# a pair RDD is generated 
responseCodeToCount = (access_logs
                       .map(lambda log: (log.response_code, 1))
                       .reduceByKey(lambda a, b : a + b)
                       .cache())
responseCodeToCountList = responseCodeToCount.take(100)
print ('Found %d response codes' % len(responseCodeToCountList))
print ('Response Code Counts: %s' % responseCodeToCountList)
assert len(responseCodeToCountList) == 7
assert sorted(responseCodeToCountList) == [(200, 940847), (302, 16244), (304, 79824), (403, 58), (404, 6185), (500, 2), (501, 17)]

labels = responseCodeToCount.map(lambda x_y: x_y[0]).collect()
print (labels)
count = access_logs.count()
fracs = responseCodeToCount.map(lambda x_y: (float(x_y[1]) / count)).collect()
print (fracs)

import matplotlib.pyplot as plt
def pie_pct_format(value):
    """ Determine the appropriate format string for the pie chart percentage label
    Args:
        value: value of the pie slice
    Returns:
        str: formated string label; if the slice is too small to fit, returns an empty string for label """
    return '' if value < 7 else '%.0f%%' % value
fig = plt.figure(figsize=(4.5, 4.5), facecolor='white', edgecolor='white')
colors = ['yellowgreen', 'lightskyblue', 'red', 'purple', 'lightcoral', 'yellow', 'green']
explode = (0.05, 0.05, 0.1, 0, 0, 0, 0)
patches, texts, autotexts = plt.pie(fracs, labels=labels, colors=colors, explode=explode, autopct=pie_pct_format, shadow=False,  startangle=125)
for text, autotext in zip(texts, autotexts):
    if autotext.get_text() == '':
        text.set_text('')  # If the slice is small to fit, don't show a text label
plt.legend(labels, loc=(0.80, -0.1), shadow=True)
display(fig)  
pass

# Create a DataFrame and visualize using display()
responseCodeToCountRow = responseCodeToCount.map(lambda x_y: Row(response_code=x_y[0], count=x_y[1]))
responseCodeToCountDF = sqlContext.createDataFrame(responseCodeToCountRow)
display(responseCodeToCountDF)

# Any hosts that has accessed the server more than 10 times.
hostCountPairTuple = access_logs.map(lambda log: (log.host, 1))
hostSum = hostCountPairTuple.reduceByKey(lambda a, b : a + b)
hostMoreThan10 = hostSum.filter(lambda s: s[1] > 10)
hostsPick20 = (hostMoreThan10
               .map(lambda s: s[0]) # s[0] is the IP address, we don't need the count 
               .take(20))
print ('Any 20 hosts that have accessed more then 10 times: %s' % hostsPick20)
# An example: [u'204.120.34.185', u'204.243.249.9', u'slip1-32.acs.ohio-state.edu', u'lapdog-14.baylor.edu', u'199.77.67.3', u'gs1.cs.ttu.edu', u'haskell.limbex.com', u'alfred.uib.no', u'146.129.66.31', u'manaus.bologna.maraut.it', u'dialup98-110.swipnet.se', u'slip-ppp02.feldspar.com', u'ad03-053.compuserve.com', u'srawlin.opsys.nwa.com', u'199.202.200.52', u'ix-den7-23.ix.netcom.com', u'151.99.247.114', u'w20-575-104.mit.edu', u'205.25.227.20', u'ns.rmc.com']


# Create an RDD with Row objects
endpoints = (access_logs
             .map(lambda log: (log.endpoint, 1))
             .reduceByKey(lambda a, b : a + b)
             .cache())
endpoint_counts_rdd = endpoints.map(lambda s: Row(endpoint = s[0], num_hits = s[1]))
endpoint_counts_schema_rdd = sqlContext.createDataFrame(endpoint_counts_rdd)

# Display a plot of the distribution of the number of hits across the endpoints.
display(endpoint_counts_schema_rdd)

topEndpoints = endpoints.takeOrdered(10, lambda s: -1 * s[1])

print ('Top Ten Endpoints: %s' % topEndpoints)
assert topEndpoints == [(u'/images/NASA-logosmall.gif', 59737), (u'/images/KSC-logosmall.gif', 50452), (u'/images/MOSAIC-logosmall.gif', 43890), (u'/images/USA-logosmall.gif', 43664), (u'/images/WORLD-logosmall.gif', 43277), (u'/images/ksclogo-medium.gif', 41336), (u'/ksc.html', 28582), (u'/history/apollo/images/apollo-logo1.gif', 26778), (u'/images/launch-logo.gif', 24755), (u'/', 20292)], 'incorrect Top Ten Endpoints'


not200 = (access_logs
          .map(lambda log: (log.response_code,log.endpoint))
          .filter(lambda x_y: x_y[0] != 200))

endpointCountPairTuple = not200.map(lambda x_y:(x_y[1],1))
endpointSum = endpointCountPairTuple.reduceByKey(lambda x,y: x+y)

topTenErrURLs = endpointSum.takeOrdered(10,key = lambda x: -x[1])
print ('Top Ten failed URLs: %s' % topTenErrURLs)


hosts = (access_logs
        .map(lambda log: (log.host,1))
        .reduceByKey(lambda x,y : x+y))
#uniqueHosts = hosts.map(lambda (x,y): x)

uniqueHostCount = hosts.count()
print ('Unique hosts: %d' % uniqueHostCount)

dayToHostPairTuple=access_logs.map(lambda log:(log.date_time.day,log.host))
#below Group the values for each key in the RDD into a single sequence.	
dayGroupedHosts=dayToHostPairTuple.groupByKey() # this line groups all the elements of the RDD with the same key into a single list: (key, [list_of_values])
# dayGroupedHosts is an rdd like: (day1, [IP1,IP2, ...]), (day2,[IP1, IP2, ...]), ...
# below v is a list and set(v) is another list with no repetition of hosts (unique hosts)
dayHostCount = dayGroupedHosts.map(lambda date_hosts: (date_hosts[0], set(date_hosts[1])))
# now count the number of unique hosts (i.e. count elements of the list >> len())
dayHostCount = dayHostCount.map(lambda date_UniqueHosts:(date_UniqueHosts[0],len(date_UniqueHosts[1])))

#dailyHostsList = dailyHosts.take(30)
# sort by increasing day of the month: day1, day2, day3, ...
dailyHosts = (dayHostCount.sortByKey().cache())
dailyHostsList = dailyHosts.take(30)
print ('Unique hosts per day: %s' % dailyHostsList)

daysWithHosts = dailyHosts.map(lambda k_v:k_v[0]).collect() # list of unique hosts
hosts = dailyHosts.map(lambda k_v:k_v[1]).collect() # list of number of count of each unique host
fig = plt.figure(figsize=(8,5), facecolor='white', edgecolor='white')
plt.axis([min(daysWithHosts), max(daysWithHosts), 0, max(hosts)+500])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Day')
plt.ylabel('# of Hosts')
plt.plot(daysWithHosts, hosts)
display(fig)  
pass

dayAndHostTuple = access_logs.map(lambda log:(log.date_time.day,log.host))

groupedByDay = dayAndHostTuple.groupByKey()

sortedByDay = groupedByDay.sortByKey() # sort by days
# sortedByDay is an rdd like: (day1, [IP1,IP2, ...]), (day2,[IP1, IP2, ...]), ...
avgDailyReqPerHost=(sortedByDay.map(lambda k_v:(k_v[0],len(k_v[1])))  
                    .join(dailyHosts)#dailyHosts is an RDD like: (day1, 2582), (day3, 3222), ... it means at day1 there are a total of 2582 UNIQUE hosts
                    .sortByKey()#after join we have rdd: (day1, (# of hosts on day1, # of unique hosts on day1)), (day2, (# of hosts on d2, # of unique hosts on d2))
                    # next get the total number of request across all hosts and divide that by the number of unique hosts
                    .map(lambda day_numHosts_numUniqHosts:(day_numHosts_numUniqHosts[0],(day_numHosts_numUniqHosts[1][0]/day_numHosts_numUniqHosts[1][1])))).cache() 
                   #.map(lambda (k,(v1,v2)):(k,(v1/v2))))
avgDailyReqPerHostList = avgDailyReqPerHost.take(30)
print ('Average number of daily requests per Hosts is %s' % avgDailyReqPerHostList)

daysWithAvg = avgDailyReqPerHost.map(lambda x_v:x_v[0]).collect()
avgs = avgDailyReqPerHost.map(lambda x_v:x_v[1]).collect()

fig = plt.figure(figsize=(8,4.8), facecolor='white', edgecolor='white')
plt.axis([0, max(daysWithAvg), 0, max(avgs)+2])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Day')
plt.ylabel('Average daily req. per host')
plt.plot(daysWithAvg, avgs)
display(fig)  
pass

badRecords = (access_logs.filter(lambda log:log.response_code==404)).cache()
print ('Found %d 404 URLs' % badRecords.count())

badEndpoints = badRecords.map(lambda log:(log.endpoint,1)).reduceByKey(lambda a,b:a+b)

badUniqueEndpoints = badEndpoints.map(lambda endpoint_count:endpoint_count[0])

badUniqueEndpointsPick40 = badUniqueEndpoints.take(40)
print ('404 URLS: %s' % badUniqueEndpointsPick40)

badEndpointsCountPairTuple = badRecords.map(lambda log:(log.endpoint,1))

badEndpointsSum = badEndpointsCountPairTuple.reduceByKey(lambda a,b:a+b)

badEndpointsTop20 = badEndpointsSum.takeOrdered(20, lambda s:-1*s[1])
print ('Top Twenty 404 URLs: %s' % badEndpointsTop20)

errHostsCountPairTuple = badRecords.map(lambda x:(x.host,1))

errHostsSum = errHostsCountPairTuple.reduceByKey(lambda a,b:a+b)

errHostsTop25 = errHostsSum.takeOrdered(25, lambda s:-1*s[1])
print ('Top 25 hosts that generated errors: %s' % errHostsTop25)

errDateCountPairTuple = badRecords.map(lambda log:(log.date_time.day,log.host)).groupByKey()

errDateSum = errDateCountPairTuple.map(lambda x_v:(x_v[0],len(x_v[1])))

errDateSorted = (errDateSum.sortByKey().cache())

errByDate = errDateSorted.collect()
print ('404 Errors by day: %s' % errByDate)

daysWithErrors404 = errDateSorted.map(lambda x_v:x_v[0]).collect()
errors404ByDay = errDateSorted.map(lambda x_v:x_v[1]).collect()

fig = plt.figure(figsize=(8,4.8), facecolor='white', edgecolor='white')
plt.axis([0, max(daysWithErrors404), 0, max(errors404ByDay)])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Day')
plt.ylabel('404 Errors')
plt.plot(daysWithErrors404, errors404ByDay)
display(fig)  
pass

topErrDate = errDateSorted.takeOrdered(5, lambda s:-1*s[1])
print ('Top Five dates for 404 requests: %s' % topErrDate)

hourCountPairTuple = badRecords.map(lambda log:(log.date_time.hour,log.response_code))
hourRecordsSum = hourCountPairTuple.groupByKey().map(lambda x_v:(x_v[0],len(x_v[1])))
hourRecordsSorted = (hourRecordsSum.sortByKey().cache())
errHourList = hourRecordsSorted.collect()
print ('Top hours for 404 requests: %s' % errHourList)

hoursWithErrors404 = hourRecordsSorted.map(lambda x_v:x_v[0]).collect()
errors404ByHours = hourRecordsSorted.map(lambda x_v:x_v[1]).collect()

fig = plt.figure(figsize=(8,4.5), facecolor='white', edgecolor='white')
plt.axis([0, max(hoursWithErrors404), 0, max(errors404ByHours)])
plt.grid(b=True, which='major', axis='y')
plt.xlabel('Hour')
plt.ylabel('404 Errors')
plt.plot(hoursWithErrors404, errors404ByHours)
display(fig)  
pass

