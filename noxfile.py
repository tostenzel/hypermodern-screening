import nox



@nox.session(python=["3.8", "3.7"])
def tests(session):

    # for running specific test files via nox
    args = session.posargs or ["--cov"]

    session.install("poetry")
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
