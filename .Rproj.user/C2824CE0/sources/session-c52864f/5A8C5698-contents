\name{GLMs with IRLS}
\alias{irls}
\title{
GLMs with IRLS
}

\description{
GLMs with IRLS.
}

\usage{
irls(y, x, type = "logistic", maxiter = 100, tol = 1e-6)
}

\arguments{
\item{y}{
A numberical vector with the response. Binary data for the binomial regression, count data for the Poisson regression and strictly positive continuous numbers for the Gamma regression.
}
\item{x}{
A numerical matrix or a vector.
}
\item{type}{
The type of regression model to perform, "logistic", "poisson" or "gamma".
}
\item{maxiter}{
The maximum number of iterations to perform.
}
\item{tol}{
The tolerance value to terminate the algorithm.
}
}

\details{
The function does logistic, Poisson and Gamma regression via the IRLS algorithm.
}

\value{
A list including:
\item{coefficients}{
The regression coefficients.
}
\item{vcov}{
The variance covariance matrix of the coefficients.
}
\item{se}{
The standard errors of the coefficients.
}
\item{phi}{
The dispersion parameter \eqn{phi} of the Gamma regression.
}
\item{deviance}{
The deviance of the regression model.
}
\item{iters}{
The number of iterations required.
}
}

\references{
McCullagh, Peter, and John A. Nelder. Generalized linear models. CRC press, USA, 2nd edition, 1989.
}

\author{
Michail Tsagris, Nikolaos Kontemeniotis and Christos Adam.
R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

\seealso{
  \code{\link{col.irls}}
}


\examples{
x <- as.matrix(iris[, 1:4])
y <- rbinom(150, 1, 0.5)
irls(y, x)
}

