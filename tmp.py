import pandas as pd

# authors = ['AaronPressman', 'AlanCrosby', 'AlexanderSmith', 'BenjaminKangLim', 'BernardHickey', 'BradDorfman', 'DarrenSchuettler', 'DavidLawder']
# cols = ['a', 's', 'd', 'f', 'g', 'h', 'e', 'w']
rows = [i for i in range(8)]
cols = [i for i in range(8)]
initial_values = [[i+j*8 for i in range(8)] for j in range(8)]
conf_matrix = pd.DataFrame(initial_values, index=rows, columns=cols)
print(conf_matrix)

for i in range(8):
    for j in range(8):
        print(conf_matrix[j][i])