try:
    import latticeseq_b2
except ImportError:
    from urllib.request import urlopen
    with urlopen('https://bitbucket.org/dnuyens/qmc-generators/raw/cb0f2fb10fa9c9f2665e41419097781b611daa1e/python/latticeseq_b2.py') as qmc_module:
        with open('latticeseq_b2.py',mode='x+b') as qmc_module_file:
            qmc_module_file.write(qmc_module.read())

    
