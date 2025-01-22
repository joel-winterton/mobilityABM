Commuting data is preprocessed in a Colab notebook from a previous project on kernel homogeneity. The original data sources are below, as well as a link to the notebook.

**Commuter flow data:**

https://www.nomisweb.co.uk/sources/census_2021_od

Using dataset ODWP01EW, LTLA geography.

**Geography data**:

https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-boundaries-gb-bfc/about

**Preprocessing notebook**:

https://colab.research.google.com/drive/1GuFEJxbzmVI5kIyNtBUAmIjcSZdUTxpn

**Patch census data**:

We just want the people who are registered on the census, so we sum down columns.