\name{Column-wise GLMs with IRLS}
\alias{col.irls}
\title{
Column-wise GLMs with IRLS
}

\description{
GLMs with IRLS.
}

\usage{
col.irls(y, x, type = "logistic", maxiter = 100, tol = 1e-6, parallel = FALSE)
}

\arguments{
\item{y}{
A numberical vector with the response. Binary data for the binomial regression, count data for the Poisson regression and strictly positive continuous numbers for the Gamma regression.
}
\item{x}{
A numerical matrix.
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
\item{parallel}{
Should the models be performed in parallel?
}
}

\details{
The function does logistic, Poisson and Gamma regression via the IRLS algorithm, for each column of x.
}

\value{
A matrix with 3 or 4 columns with the \eqn{\alpha} (constant) and \eqn{\beta} parameters, the deviance and the \eqn{\phi} (dispersion) parameter in case of Gamma regression.
}

\references{
McCullagh, Peter, and John A. Nelder. Generalized linear models. CRC press, USA, 2nd edition, 1989.
}

\author{
Michail Tsagris, Nikolaos Kontemeniotis and Christos Adam.
R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

\seealso{
  \code{\link{irls}}
}

\examples{
x <- as.matrix(iris[, 1:4])
y <- rbinom(150, 1, 0.5)
col.irls(y, x)
}

