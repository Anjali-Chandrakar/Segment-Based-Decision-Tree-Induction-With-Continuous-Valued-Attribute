from hybrid import scores,filename
from c4_5 import scores as s2
import matplotlib.pyplot as plt
import numpy as np

index = np.arange(len(s2))
width = 0.35
plt.bar( index,scores,width, color = 'green',label='segment+c4.5')
plt.bar( index+width,s2,width, color = 'blue',label='c4.5')
plt.title(filename)
plt.xlabel('k-folds')
plt.ylabel('Accuracy')
plt.xticks(index + width, ('1', '2', '3', '4','5','6','7','8','9','10'))
plt.show()
