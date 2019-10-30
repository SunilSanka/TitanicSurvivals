package titanic;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.MaxAbsScalerModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.util.kvstore.LevelDB;

public class Titanic {

	/* Convert to Double */
	public static UDF1 convertToDouble = new UDF1<Integer, Double>() {
		public Double call(Integer SibSp) throws Exception {
			return new Double(SibSp);
		}
	};
	
	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		// Creating Spark Session
		SparkSession sparkSession = SparkSession.builder().appName("Titanic").master("local[*]").getOrCreate();

		// Traing Data
		Dataset<Row> input = sparkSession.read().option("header", true).option("inferschema", true).csv("data/train.csv");
		Dataset<Row> testdata = sparkSession.read().option("header", true).option("inferschema", true).csv("data/test.csv");

		// Register UDF
		sparkSession.udf().register("convertToDouble", convertToDouble,DataTypes.DoubleType);

		//Select required rows
		input = input.select(col("SibSp"),col("Parch"),col("Age"),col("Pclass"),col("Fare"),col("Sex"),col("Survived"));

		//Converting the datatype to double
		input = input.withColumn("SibSp-Dbl", callUDF("convertToDouble", col("SibSp")));
		input = input.withColumn("Parch-Dbl", callUDF("convertToDouble", col("Parch")));
		input = input.withColumn("Pclass-Dbl", callUDF("convertToDouble", col("Pclass")));

		//Imputer
		Imputer imputer = new Imputer()
				.setInputCols(new String[] {"SibSp-Dbl","Parch-Dbl","Age","Pclass-Dbl","Fare"})
				.setOutputCols(new String[] { "SibSp-Out","Parch-Out","Age-Out","Pclass-Out","Fare-Out"});

		input = imputer.fit(input).transform(input);
		
		// Fetching Family Size
		input = input.withColumn("familySize", input.col("SibSp-Out").plus(col("Parch-Out")));
		
		// Multiplying Age X pClass
		input = input.withColumn("ageClass", col("Age-Out").multiply(col("Pclass-Out")));
		
		// Calculating Fare_Per_Person
		input = input.withColumn("farePerPerson",input.col("Fare-Out").divide(input.col("familySize").plus(1)));
		
		// StringIndexer
		StringIndexer genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("Sex-Out");
		input = genderIndexer.fit(input).transform(input);
		
		// Final Input Dataframe
		System.out.println("*********************************** Selected Input Data ***********************************");
		Dataset<Row> selectedInput = input.select(input.col("Survived").as("label"), input.col("familySize"),input.col("ageClass"), input.col("farePerPerson"), input.col("Sex-Out"));
		selectedInput.show();
		
		// Assembling the features in the dataFrame as Dense Vector
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] { "familySize", "ageClass", "farePerPerson", "Sex-Out" })
				.setOutputCol("features");

		// Seleting label and features
		Dataset<Row> finalInput = assembler.transform(selectedInput).select("label", "features");

		/********************* Decision Tree Classifier Model Preparation ***********************************/
		// Decision Tree
		DecisionTreeClassifier dt = new DecisionTreeClassifier()
				.setFeaturesCol("features")
				.setLabelCol("label")
				.setSeed(1L);

		// Decision Tree Model
		DecisionTreeClassificationModel dtModel = dt.fit(finalInput);

		/********************* Random Forest Model Preparation ***********************************/
		// Random Forest
		RandomForestClassifier rf = new RandomForestClassifier()
				.setFeaturesCol("features")
				.setLabelCol("label")
				.setSeed(1L);

		// Random Forest Model
		RandomForestClassificationModel rfModel = rf.fit(finalInput);

		/******************** Model Evalautions ***********************************/
		System.out.println("*********************************** Decision Tree Classification Model ***********************************");
		// Decision Tree Classification Model
		Dataset<Row> trainDtPredictions = dtModel.transform(finalInput);
		trainDtPredictions.show(10);

		System.out.println("*********************************** Random Forest Classification Model ***********************************");
		// Random Forest Classification Model
		Dataset<Row> trainRfPredictions = rfModel.transform(finalInput);
		trainRfPredictions.show(10);

		// View confusion matrix
		 /*
		 System.out.println("*********************************** Confusion Matrix ***********************************");
		 trainDtPredictions.groupBy(col("label"), col("prediction")).count().show();
		 trainRfPredictions.groupBy(col("label"), col("prediction")).count().show(); */

		// Accuracy computation
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("accuracy");

		// Evaluate Decision Tree Classification Model
		System.out.println("*********************************** Decision Tree Evaluation  ***********************************");

		double accuracyDT = evaluator.evaluate(trainDtPredictions);
			System.out.println("Accuracy = " + Math.round(accuracyDT * 100) + " %");
			System.out.println("Test Error =" + (1.0 - accuracyDT));

		// Evaluate Random Forest Classification Model
		System.out.println("*********************************** Decision Tree Evaluation  ***********************************");
		double accuracyRF = evaluator.evaluate(trainRfPredictions);
			System.out.println("Accuracy = " + Math.round(accuracyRF * 100) + " %");
			System.out.println("Test Error = " + (1.0 - accuracyRF));

		/******************** Test Data Preparation ***********************************/
		System.out.println("*********************************** Preparing Test Data  ***********************************");
		
		testdata = testdata.select(col("PassengerId"),col("SibSp"),col("Parch"),col("Age"),col("Pclass"),col("Fare"),col("Sex"));
		
		
		//Converting the datatype to double
		testdata = testdata.withColumn("SibSp-Dbl", callUDF("convertToDouble", col("SibSp")));
		testdata = testdata.withColumn("Parch-Dbl", callUDF("convertToDouble", col("Parch")));
		testdata = testdata.withColumn("Pclass-Dbl", callUDF("convertToDouble", col("Pclass")));
		testdata.printSchema();
		
		//Imputer
		testdata = imputer.fit(testdata).transform(testdata);
		
		// Fetching Family Size
		testdata = testdata.withColumn("familySize", testdata.col("SibSp-Out").plus(col("Parch-Out")));
				
		// Multiplying Age X pClass
		testdata = testdata.withColumn("ageClass", col("Age-Out").multiply(col("Pclass-Out")));
			
		// Calculating Fare_Per_Person
		testdata = testdata.withColumn("farePerPerson",testdata.col("Fare-Out").divide(testdata.col("familySize").plus(1)));
				
		// StringIndexer
		testdata = genderIndexer.fit(testdata).transform(testdata);

		// Final Input Dataframe
		System.out.println("*********************************** Selected Input Data ***********************************");
		Dataset<Row> selectedTestInput = testdata.select(testdata.col("PassengerId"),testdata.col("familySize"),testdata.col("ageClass"), testdata.col("farePerPerson"), testdata.col("Sex-Out"));
		selectedTestInput.show();
				
		// Vector Assembler
		Dataset<Row> assembledTestData = assembler.transform(selectedTestInput);
		
		// Decision Tree Predictions - Test Data
		/******************** Decision Tree ***********************************/
		System.out.println("*********************************** Decision Tree Classifier - Test Data ***********************************");
		Dataset<Row> predictionsDt = dtModel.transform(assembledTestData);
		predictionsDt.show();

		/******************** Random Forest ***********************************/
		System.out.println("*********************************** Random Forest Classification - Test Data ***********************************");
		Dataset<Row> predictionsRf = rfModel.transform(assembledTestData);
		predictionsRf.show();

		// Saving Random Forest
		Dataset<Row> results = predictionsRf.select(col("PassengerId"),	col("prediction").as("Survived").cast(DataTypes.IntegerType));
		results.show();
		results.write().option("header", "true").csv("C:\\Personal\\PGPBD\\Titanic\\results\\gender_submission.csv");
	}
}
