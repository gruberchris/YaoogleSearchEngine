using Lucene.Net.Analysis.En;
using Lucene.Net.Util;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SearchEngineCSharp
{
    class Program
    {
        // Sample documents
        static List<string> documents =
        [
            "Vice President Kamala Harris’ campaign laid out what it sees as her path to victory in Pennsylvania in a memo shared exclusively with NBC News ahead of Monday night’s rally in bellwether Erie County.\n\nThe Harris team pointed to polls showing Harris, the Democratic presidential nominee, having made gains in the battleground state’s suburbs — which it dubbed “our own mini ‘blue wall’” in Pennsylvania — compared with President Joe Biden’s 2020 performance there.",
            "Canada announced the expulsion of six Indian diplomats Monday, including the high commissioner, after the police accused agents of the Indian government of being linked to homicides, harassment and other “acts of violence” against Sikh separatists in the country.\n\nCanada’s Minister of Foreign Affairs Mélanie Joly said in a Monday statement that the decision to expel the diplomats “was made with great consideration and only after (Canadian police) gathered ample, clear and concrete evidence which identified six individuals as persons of interest in the Nijjar case.”",
            "Federal disaster workers paused and then changed some of their hurricane-recovery efforts in North Carolina, including abandoning door-to-door visits, after receiving threats that they could be targeted by a militia, officials said, as the government response to Helene is targeted by runaway disinformation.\n\nThe threats emerged over the weekend. The Rutherford County Sheriff’s Office said in a statement Monday that it received a call Saturday about a man with an assault rifle who made a comment “about possibly harming” employees of the Federal Emergency Management Agency working in the hard-hit areas of Lake Lure and Chimney Rock, in the North Carolina mountains.",
            "With the release of Vessel of Hatred, Diablo IV has seen its most significant changes since its original launch in June 2023. Adding a completely new region, Nahantu, along with a wealth of added characters and modes and a brand-new story, the expansion pack is the very definition of a game changer. But it goes even further than that, bringing in entirely new ways to upgrade items, a revamp of the World Tiers, new animal-morphing classes, and a new level cap. However, we meet change without fear, offering a litany of guides to get you up to speed.\n\nFor instance, Diablo IV now has NPC Mercenaries you can hire to come with you on your brawling, but you’re only going to find them by following a specific series of quests. Then there are those Spiritborn classes that let you pick between being able to possess the powers of a Jaguar, Eagle, Gorilla, or Centipede…Wait, centipede? No, we’ve checked, that’s definitely correct—and apparently one of the best choices for end-game content.",
            "The benchmark Selic will hit 11% at year-end 2025, up from the prior estimate of 10.75%, according to a weekly central bank survey published Monday. Estimates for borrowing costs this December stayed unchanged at 11.75%.\n\nLatin America’s largest economy has proved more resilient than expected during most of this year, prompting policymakers to kick off a tightening cycle in September by lifting borrowing costs to 10.75%. Low unemployment, a weaker real and an increase in families’ disposable income due in part to higher government transfers are spurring fears of inflation pressures.\n\n“Brazil’s economy continues to give signals of more resilience and stronger activity,” Gabriel Galipolo, the central bank’s current monetary policy director and next governor, said at an event in Sao Paulo later on Monday."
        ];

        static void Main(string[] args)
        {
            // Preprocess the documents
            var processedDocs = documents.Select(doc => Preprocess(doc)).ToList();

            // Build the TF-IDF matrix
            var tfidf = new TfIdfVectorizer();
            tfidf.Fit(processedDocs);
            var tfidfMatrix = tfidf.Transform(processedDocs);

            // Prompt user for search query
            Console.Write("Enter your search query: ");
            var query = Console.ReadLine();

            // Preprocess the query
            var processedQuery = Preprocess(query);

            // Transform the query into TF-IDF vector
            var queryVector = tfidf.Transform(new List<List<string>> { processedQuery }).FirstOrDefault();
            
            // Compute cosine similarity between query and documents
            var similarities = ComputeCosineSimilarities(queryVector, tfidfMatrix);

            // Get top matching documents
            var topResults = GetTopResults(similarities, documents, topN: 5);

            // Display the results
            if (topResults.Any())
            {
                Console.WriteLine("\nTop search results:");
                foreach (var result in topResults)
                {
                    Console.WriteLine($"\nScore: {result.Score:F4} | Document: {result.Document}");
                }
            }
            else
            {
                Console.WriteLine("No matching documents found.");
            }
        }

        // Preprocess text: tokenize, remove stop words, stem
        static List<string> Preprocess(string text)
        {
            var analyzer = new EnglishAnalyzer(LuceneVersion.LUCENE_48);

            var tokenStream = analyzer.GetTokenStream("field", text);
            tokenStream.Reset();

            var tokens = new List<string>();
            while (tokenStream.IncrementToken())
            {
                var termAttribute = tokenStream.GetAttribute<Lucene.Net.Analysis.TokenAttributes.ICharTermAttribute>();
                tokens.Add(termAttribute.ToString());
            }

            tokenStream.End();
            tokenStream.Dispose();

            return tokens;
        }

        // Compute cosine similarities between the query vector and document vectors
        static List<double> ComputeCosineSimilarities(Vector<double> queryVector, List<Vector<double>> tfidfMatrix)
        {
            var similarities = new List<double>();

            foreach (var docVector in tfidfMatrix)
            {
                var similarity = CosineSimilarity(queryVector, docVector);
                similarities.Add(similarity);
            }

            return similarities;
        }

        // Cosine similarity between two vectors
        static double CosineSimilarity(Vector<double> vectorA, Vector<double> vectorB)
        {
            if (vectorA.L2Norm() <= 0 || vectorB.L2Norm() <= 0)
                return 0;

            return vectorA.DotProduct(vectorB) / (vectorA.L2Norm() * vectorB.L2Norm());
        }

        // Get top N results based on similarity scores
        static List<(string Document, double Score)> GetTopResults(List<double> similarities, List<string> documents, int topN)
        {
            var results = similarities
                .Select((score, index) => new { Score = score, Document = documents[index] })
                .Where(x => x.Score > 0)
                .OrderByDescending(x => x.Score)
                .Take(topN)
                .Select(x => (x.Document, x.Score))
                .ToList();

            return results;
        }
    }

    // TF-IDF Vectorizer class
    public class TfIdfVectorizer
    {
        private Dictionary<string, int> vocabulary = new Dictionary<string, int>();
        private Dictionary<string, double> idfScores = new Dictionary<string, double>();
        private int totalDocuments;

        public void Fit(List<List<string>> tokenizedDocuments)
        {
            totalDocuments = tokenizedDocuments.Count;

            // Build vocabulary
            foreach (var doc in tokenizedDocuments)
            {
                foreach (var token in doc.Distinct())
                {
                    if (!vocabulary.ContainsKey(token))
                    {
                        vocabulary[token] = vocabulary.Count;
                    }
                }
            }

            // Calculate IDF scores
            foreach (var term in vocabulary.Keys)
            {
                int docCount = tokenizedDocuments.Count(doc => doc.Contains(term));
                idfScores[term] = Math.Log((double)totalDocuments / (1 + docCount));
            }
        }

        public List<Vector<double>> Transform(List<List<string>> tokenizedDocuments)
        {
            var tfidfVectors = new List<Vector<double>>();
            foreach (var doc in tokenizedDocuments)
            {
                var tfidfVector = new double[vocabulary.Count];

                // Calculate term frequencies
                var termFrequencies = doc.GroupBy(t => t)
                                         .ToDictionary(g => g.Key, g => g.Count());

                foreach (var term in termFrequencies.Keys)
                {
                    if (vocabulary.ContainsKey(term))
                    {
                        int index = vocabulary[term];
                        double tf = (double)termFrequencies[term] / doc.Count;
                        double idf = idfScores[term];
                        tfidfVector[index] = tf * idf;
                    }
                }

                tfidfVectors.Add(DenseVector.OfArray(tfidfVector));
            }

            return tfidfVectors;
        }
    }
}
