# Paieskos-algoritmas
LD1. Paieškos algoritmų tyrimas

## Repositorija, laboratoriniu darbu ginimui

Ši sistema naudoja dirbtinio intelekto pagrindu veikiančius teksto embedding modelius straipsnių analizei ir paieškai. Naudojant SentenceTransformer modelį straipsniai paverčiami į vektorius, o KMeans algoritmas suskirsto juos į 5 temines grupes. Raktažodžiai išgaunami naudojant KeyBERT, kuris remiasi transformer architektūra ir semantiniu panašumu. Sistema leidžia vartotojui atlikti semantinę paiešką pagal įvestą frazę, naudojant cosine similarity metodą. Vartotojo sąsaja sukurta naudojant Streamlit.

### Neitraukiami dataset ir artifact failai, dėl failų dydžio limito (100MB )