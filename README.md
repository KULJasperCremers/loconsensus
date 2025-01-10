# loconsensus
python version <3.13

pip install ./requirements

loconsensus => ./loconsensus
- subjects.pkl contains all the preprocessed data of the PAMAP2 dataset
- ts_list has to be a list of numpy arrays with shape (len, dim), e.g. (5000,2) for a 2-dimensional timeseries with 5000 datapoints.
```
import loconsensus.loconsensus as loconsensus

l_min = 15
l_max = 30
rho = 0.8
nb = None

motifs = loconsensus.apply_loconsensus(ts_list, l_min, l_max, rho, nb)
```


experiment  => ./experiments
- runtimes.pkl contains the output from the experiement

usecase     => ./usecase
- cm_single.pkl contains all the consensus motifs from the first usecase setup.
- cm_double.pkl contains all the consensus motifs from the second usecase setup.