
// Generated C# class for ML.NET consumption (Pickle via Python.NET)
// Model: DARF-Classifier.pkl

// IMPORTANT: Requires Python.NET package
// Install: Install-Package Python.NET

using Python.Runtime;

public class ModelInput
{
    public double Age { get; set; }
    public double Gender { get; set; }
    public double BMI { get; set; }
    public double Family_History { get; set; }
    public double Smoking { get; set; }
    public double Alcohol_Consumption { get; set; }
    public double Physical_Activity { get; set; }
    public double Hormone_Therapy { get; set; }
    public double Menopause_Status { get; set; }
    public double Genetic_Mutation { get; set; }
    public double Tumor_Size_cm { get; set; }
    public double Lymph_Node_Involvement { get; set; }
    public double Mammogram_Result { get; set; }
    public double Biopsy_Result { get; set; }
    public double Cancer_Stage { get; set; }
    public double Blood_Pressure { get; set; }
    public double Cholesterol { get; set; }
    public double Diabetes { get; set; }
    public double Exercise_Days_Per_Week { get; set; }
    public double Breastfeeding_History { get; set; }
    public double Annual_Income_USD { get; set; }

}

public class ModelOutput
{
    public double[] PredictedLabel { get; set; }
}

public class PickleModelPredictor
{
    private dynamic _model;
    private dynamic _scaler;
    private dynamic _np;
    
    public PickleModelPredictor(string modelPath)
    {
        using (Py.GIL())
        {
            var pickle = Py.Import("pickle");
            using var file = Py.Import("builtins").open(modelPath, "rb");
            _model = pickle.load(file);
            _np = Py.Import("numpy");
        }
    }
    
    public ModelOutput Predict(ModelInput input)
    {
        using (Py.GIL())
        {
            // Create numpy array from input
            var features = new double[] {
        input.Age,
        input.Gender,
        input.BMI,
        input.Family_History,
        input.Smoking,
        input.Alcohol_Consumption,
        input.Physical_Activity,
        input.Hormone_Therapy,
        input.Menopause_Status,
        input.Genetic_Mutation,
        input.Tumor_Size_cm,
        input.Lymph_Node_Involvement,
        input.Mammogram_Result,
        input.Biopsy_Result,
        input.Cancer_Stage,
        input.Blood_Pressure,
        input.Cholesterol,
        input.Diabetes,
        input.Exercise_Days_Per_Week,
        input.Breastfeeding_History,
        input.Annual_Income_USD,

            };
            var arr = _np.array(features).reshape(1, -1);
            
            // Make prediction
            var result = _model.predict(arr);
            
            // Convert to .NET array
            var output = new ModelOutput();
            var pyList = result.tolist();
            output.PredictedLabel = ((PyList)pyList).Select(x => (double)x).ToArray();
            
            return output;
        }
    }
}

// Usage:
// var predictor = new PickleModelPredictor(@"models\DARF-Classifier.pkl");
// var input = new ModelInput { ... };
// var output = predictor.Predict(input);
