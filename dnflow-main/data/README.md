# Data placement

Raw datasets are not tracked because redistribution terms differ by source and
several files are too large for a source repository. Obtain each dataset from its
official source and comply with its license.

Place the prepared files as follows:

```text
data/
  adult/{real.csv,test.csv}
  default/{real.csv,test.csv}
  shoppers/{real.csv,test.csv}
  magic/{real.csv,test.csv}
  beijing/{real.csv,test.csv}
  news/{real.csv,test.csv}
  bank_marketing/{real.csv,test.csv}
  bike_sharing/{real.csv,test.csv}
```

The corresponding schemas are in `data/metadata/`. `real.csv` is the development
pool from which the deterministic training/validation split is made; `test.csv`
is held out. Do not fit preprocessing transforms on `test.csv`.

The eight source records are available from the UCI Machine Learning Repository:

- Adult: https://archive.ics.uci.edu/dataset/2/adult
- Default: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Shoppers: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Magic: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
- Beijing PM2.5: https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
- Online News Popularity: https://archive.ics.uci.edu/dataset/332/online+news+popularity
- Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Bike Sharing: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

The UCI records identify these datasets as CC BY 4.0. Attribution and dataset
citations are still required if raw or derived data is redistributed. This code
repository therefore ignores the CSV files by default. Before public release,
verify that the prepared splits derive from these exact records and add checksums.
