import nox



@nox.session(python=["3.8", "3.7"])
def tests(session):
    session.install("poetry")
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")
