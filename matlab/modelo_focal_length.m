%% Model Real Heights Rsq = 0.9959 y = 164.47x+135.23
a=164.47;
b=135.23;

heightsadults= [1.57	1.7	1.68	1.67	1.6	1.6	1.63 1.76 1.69];
newfocal_adults = a*heightsadults + b;

heightschildren = [1.1901 1.2255];
newfocal_children = a*heightschildren + b;