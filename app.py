from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy as scipy


app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate random dataset X
    X = np.random.uniform(0, 1, N)
    X = X.reshape(-1, 1)

    # Generate random dataset Y
    error = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X.flatten() + mu + error

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data points')
    plt.plot(X, model.predict(X), color='red', label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Regression Line: Y = {slope:.2f} * X + {intercept:.2f}")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # Generate simulated datasets
        X_sim = np.random.uniform(0, 1, N).reshape(-1, 1)
        error_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim.flatten() + mu + error_sim

        # Fit linear regression to simulated data
        sim_model = LinearRegression()
        sim_model.fit(X_sim, Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.7, label='Slopes', color='blue')
    plt.hist(intercepts, bins=20, alpha=0.7, label='Intercepts', color='green')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title("Histogram of Slopes and Intercepts")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.sum(np.abs(slopes) >= np.abs(slope)) / S
    intercept_extreme = np.sum(np.abs(intercepts) >= np.abs(intercept)) / S

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.sum(simulated_stats >= observed_stat) / S
    elif test_type == "<":
        p_value = np.sum(simulated_stats <= observed_stat) / S
    elif test_type == "≠":
        p_value = np.sum(np.abs(simulated_stats) >= np.abs(observed_stat)) / S
    else:
        p_value = None

    # Set fun_message if p_value is very small
    fun_message = "Wow! This is a very rare result, indicating strong evidence against the null hypothesis!" if p_value is not None and p_value <= 0.0001 else None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=20, alpha=0.7, color='blue', label='Simulated Statistics')
    plt.axvline(x=observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.axvline(x=hypothesized_value, color='green', linestyle='dotted', linewidth=2, label='Hypothesized Value')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    # Calculate confidence interval for the parameter estimate
    t_value = scipy.stats.t.ppf(1.0 - (1.0 - confidence_level/100) / 2.0, df=S - 1)
    margin_of_error = t_value * (std_estimate / np.sqrt(S))
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot the individual estimates as gray points and confidence interval
    plot4_path = "static/plot4.png"
    plt.figure()
    plt.scatter(range(len(estimates)), estimates, color='gray', alpha=0.5, label='Estimates')
    plt.axhline(y=mean_estimate, color='blue', linestyle='-', label='Mean Estimate')
    plt.axhline(y=ci_lower, color='red', linestyle='--', label='Confidence Interval Lower Bound')
    plt.axhline(y=ci_upper, color='red', linestyle='--', label='Confidence Interval Upper Bound')
    plt.axhline(y=true_param, color='green', linestyle='-.', label='True Parameter')
    plt.xlabel('Simulation Index')
    plt.ylabel('Estimate Value')
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )



if __name__ == "__main__":
    app.run(debug=True, port=3000)
