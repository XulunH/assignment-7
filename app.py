from flask import Flask, render_template, request, url_for, session
from flask_session import Session  # Import Flask-Session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t  # Import t-distribution for confidence intervals

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

# Configure server-side session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)  # Initialize the session with the app


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    error_term = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error_term

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    Y_pred = model.predict(X_reshaped)
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data')
    plt.plot(X, Y_pred, color='red', label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Regression Line: Y = {slope:.2f} * X + {intercept:.2f}")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model.fit(X_sim_reshaped, Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # TODO 8: Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(slopes, bins=30, color='blue', edgecolor='black')
    plt.axvline(x=slope, color='red', linestyle='dashed', linewidth=2, label='Observed Slope')
    plt.title('Histogram of Slopes')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(intercepts, bins=30, color='green', edgecolor='black')
    plt.axvline(x=intercept, color='red', linestyle='dashed', linewidth=2, label='Observed Intercept')
    plt.title('Histogram of Intercepts')
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slopes_array = np.array(slopes)
    intercepts_array = np.array(intercepts)

    slope_more_extreme = np.mean(np.abs(slopes_array - beta1) >= np.abs(slope - beta1))
    intercept_extreme = np.mean(np.abs(intercepts_array - beta0) >= np.abs(intercept - beta0))

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
    # This route handles data generation (same as above)
    return index()


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

    # TODO 10: Calculate p-value based on test type
    if test_type == "greater":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "less":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "not equal":
        p_value = 2 * min(
            np.mean(simulated_stats >= observed_stat), np.mean(simulated_stats <= observed_stat)
        )
    else:
        p_value = None  # Invalid test type

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value is not None and p_value <= 0.0001:
        fun_message = "Wow! You've found a p-value less than or equal to 0.0001!"
    else:
        fun_message = None

    # TODO 12: Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=30, color='gray', edgecolor='black')
    plt.axvline(
        observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic'
    )
    plt.axvline(
        hypothesized_value,
        color='blue',
        linestyle='dashed',
        linewidth=2,
        label='Hypothesized Value',
    )
    plt.title(f'Histogram of Simulated {parameter.capitalize()}s')
    plt.xlabel(parameter.capitalize())
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
        # TODO 13: Uncomment the following lines when implemented
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

    # TODO 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # TODO 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level
    degrees_of_freedom = len(estimates) - 1
    alpha = 1 - confidence_level / 100
    t_crit = t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    standard_error = std_estimate / np.sqrt(len(estimates))
    margin_of_error = t_crit * standard_error
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # TODO 16: Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # TODO 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))

    # Plot individual estimates as gray points at y=1
    plt.scatter(estimates, np.ones_like(estimates), color='gray', alpha=0.5, label='Simulated Estimates')

    # Plot mean estimate
    mean_color = 'green' if includes_true else 'red'
    plt.scatter(mean_estimate, 1, color=mean_color, s=100, label='Mean Estimate')

    # Plot confidence interval as a horizontal line
    plt.hlines(y=1, xmin=ci_lower, xmax=ci_upper, colors='blue', linestyles='solid', linewidth=2, label='Confidence Interval')

    # Plot true parameter value as a vertical line
    plt.axvline(x=true_param, color='black', linestyle='dashed', linewidth=2, label='True Parameter Value')

    plt.ylim(0.9, 1.1)  # Narrow y-axis to focus on the horizontal line
    plt.yticks([])  # Remove y-axis ticks
    plt.xlabel(parameter.capitalize())
    plt.title(f'Confidence Interval for {parameter.capitalize()} at {confidence_level}% Level')
    plt.legend()
    plt.tight_layout()
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
    app.run(debug=True)
