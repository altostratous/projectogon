import os
import pprint
import re
from collections import defaultdict

status_translator = {
    'Failed': 'Correct',
    'not considered, incorrectly': 'Wrong',
    'Error': 'Correct',

}

data = defaultdict(lambda: defaultdict(dict))

for technique in os.listdir('results'):
    for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        filename = 'results/{}/{}.txt'.format(technique, epsilon)
        for line in open(filename):
            image_id = None
            status = None
            regex = re.compile('img (\d+) (.*)')
            match = regex.match(line.strip('\n'))
            if match:
                image_id, status = match.groups()
            regex = re.compile('Image: (\d+)\tStatus: ([^ ]*?)\tDuration: \d+')
            match = regex.match(line.strip('\n'))
            if match:
                image_id, status = match.groups()
            if image_id:
                status = status_translator.get(status, status.split(' ')[0])
                image_id = int(image_id)
                data[technique][epsilon][image_id] = status

for technique in os.listdir('results'):
    for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        assert len(data[technique][epsilon]) == 100


comparison_data = defaultdict(lambda: defaultdict(int))

for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
    for image_id in range(100):
        two_relu = data['2relu'][epsilon][image_id] == 'Verified'
        projectagon = data['projectagon'][epsilon][image_id] == 'Verified'

        if projectagon:
            comparison_data['projectagon'][epsilon] += 1
        if two_relu:
            comparison_data['2relu'][epsilon] += 1

        if two_relu and projectagon:
            comparison_data['common'][epsilon] += 1
        else:
            if projectagon:
                comparison_data['projectagon_only'][epsilon] += 1
            if two_relu:
                comparison_data['2relu_only'][epsilon] += 1

print('epsilon', '2relu', 'projectagon')
for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
    print(epsilon, end=' ')
    for technique in ('2relu', 'projectagon'):
        print(comparison_data[technique][epsilon], end=' ')
    print()

print()

print('epsilon', '2relu', 'projectagon')
for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
    print(epsilon, end=' ')
    for technique in ('common', '2relu_only', 'projectagon_only'):
        print(comparison_data[technique][epsilon], end=' ')
    print()
