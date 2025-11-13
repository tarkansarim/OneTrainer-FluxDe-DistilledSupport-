import sys
print('Python path:')
for p in sys.path[:3]:
    print(p)

import mgds
print('mgds path:', mgds.__file__)

import mgds.pipelineModules.CollectPaths as cp
print('CollectPaths path:', cp.__file__)
print('Has _debug method:', hasattr(cp.CollectPaths, '_debug'))

# Check if start method has our debug prints
import inspect
source = inspect.getsource(cp.CollectPaths.start)
print('start method has debug prints:', '[CollectPaths Debug]' in source)
