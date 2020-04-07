# Tests

## Idé

Vi bør teste det meste kode, vi skriver, fordi sådan nogle projekter her tit kan sidde fast i små fejl. 

## Implementation

* Vi bruger `pytest`, som kan installeres ved `pip install -U pytest`, se [dokumentationen](https://docs.pytest.org/en/latest/getting-started.html#our-first-test-run).
* Hver fil i `src` bør have en tilsvarende fil i `tests`, som tester de fleste af funktionerne i filen. Denne skal navngives `test_"scriptnavn".py`.
* Dette test-script skal indeholder funktioner, der starter med `test_` eller klasser, der starter med `Test`, så `pytest` kan opdage dem. Disse funktioner skal så indeholde `assert`-udtryk, som der skal testes for.
* Pga. sti-struktur skal denne fil altid begynde med følgende, hvorefter man bare kan importere sin fil.
```python
import sys, os
sys.path.append(os.path.join(sys.path[0], '..', 'src'))
```


