// See https://aka.ms/new-console-template for more information
using Microsoft.Data.Sqlite;
using oostfraeiskorg;
using System.Collections.Generic;
using System.Data;
using System.Reflection;
using System.Transactions;
using static System.Net.Mime.MediaTypeNames;

const int MAX_TEXT_LENGTH = 300;
const int MANUAL_DATASET_MUL = 1;
const int SQL_SEED = 42; // Set your desired seed value


static string MapPath(string path)
{
    return Path.Combine(System.IO.Path.GetDirectoryName(Assembly.GetEntryAssembly().Location).ToString(), path);
}

String[][] InsertIntoTemplate(string eastfrisian, string german, String[][] templates)
{
    List<String> frsSentencesList = new List<String>();
    List<String> gerSentencesList = new List<String>();

    foreach (String[] template in templates)
    {
        eastfrisian = eastfrisian[0].ToString().ToUpper() + eastfrisian.Substring(1);
        frsSentencesList.Add(template[0].Split("[WORD]")[0] + eastfrisian + template[0].Split("[WORD]")[1]);
        gerSentencesList.Add(template[1].Split("[WORD]")[0] + german + template[1].Split("[WORD]")[1]);
    }

    String[] frsSentences = frsSentencesList.ToArray();
    String[] gerSentences = gerSentencesList.ToArray();

    return new String[][] { frsSentences, gerSentences };
}

String[][] InsertIntoTemplateArticle(string article, string eastfrisian, string german, String[][] templates)
{
    eastfrisian = article[0].ToString().ToUpper() + article.Substring(1) + " " + eastfrisian;

    string gerArticle = article.Equals("däi") ? "Der" : "Das";
    german = gerArticle + " " + german;

    List<String> frsSentencesList = new List<String>();
    List<String> gerSentencesList = new List<String>();

    foreach (String[] template in templates)
    {
        frsSentencesList.Add(template[0].Split("[WORD]")[0] + eastfrisian + template[0].Split("[WORD]")[1]);
        gerSentencesList.Add(template[1].Split("[WORD]")[0] + german + template[1].Split("[WORD]")[1]);
    }

    String[] frsSentences = frsSentencesList.ToArray();
    String[] gerSentences = gerSentencesList.ToArray();

    return new String[][] { frsSentences, gerSentences };
}

static string FilterAbbreviation(string str, string abbreviation)
{
    if (str.Split(abbreviation).Length > 1)
    {
        str = str.Split(abbreviation)[1];
    }
    return str;
}

static string FilterAllAbbreviations(string str)
{
    String[] abbreviations = new string[] { "geogr.: ", "zool.: ", "weller.: ", "fig.: ",
                                            "Asrf.: ", "fehnkult.: ", "kinderr.: ", "räts.: ",
                                            "zungenbr.: ", "scherzh.: ", "imk.: ", "friesensp.: ",
                                            "botan.: ", "myk.: ", "naut.: ","ziegel.: ",
                                            "kinderspr.: ", "med.: ", "geh.: ", "ugs.: ",
                                            "vulg.: ", "fis.: ", "iron.: ", "anatom.: ",
                                            "schimpfw.: ", "chem.: ", "scherz.: ", "landw.: ",
                                            "deichbaul.: ", "mühlenkult.: ", "abw.: ", "Nn.: ",
                                            "m. Vn.: ", "w. Vn.: ", "wn. Vn.: ", "zool: ", "Tr.: ",
                                            "m. V.: ", "maurerh.: ", "schimfpw.: ", "mod.: ", "m. V.: ",
                                            "Asrf: ", "tischlerh.: ", "schimpfwort.: ", "m.: ",
                                            "mühlenkult.:: ", "phys.: ", "m. Vn.n: ", "w. Vn: ",
                                            "deichbaul..: ", "anaton.: ", "kindersp.: ", "w. Vm.: ",
                                            "m.V.: "};
    foreach (var abbreviation in abbreviations)
    {
        str = FilterAbbreviation(str, abbreviation);
    }
    return str;
}

static bool IsToBeFiltered(string str)
{
    return str.Length > 8 && str.ToLower().Contains("jööed") || str.ToLower().Contains("jööden") || str.Length <= 3;
}

String[] textAbbreviations = new string[] { "bzw" };

String[][] SplitSentences(string eastfrisian, string german)
{
    List<String> frsSentencesList = new List<String>(eastfrisian.Split(". "));
    List<String> gerSentencesList = new List<String>(german.Split(". "));

    int size = Math.Min(frsSentencesList.Count, gerSentencesList.Count);

    if (german.Length > MAX_TEXT_LENGTH)
    {
        for (int i = 0; i < size; i++)
        {
            while (gerSentencesList.Count > i + 1 && gerSentencesList[i].Length >= 3 && textAbbreviations.Contains(gerSentencesList[i].Substring(gerSentencesList[i].Length - 3, 3)))
            {
                gerSentencesList[i] += ". " + gerSentencesList[i + 1];
                gerSentencesList.RemoveAt(i + 1);
            }

            if (i < size - 1)
            {
                frsSentencesList[i] += ".";
                gerSentencesList[i] += ".";
            }
        }
    }
    else
    {
        size = 1;
        frsSentencesList = new List<String>();
        gerSentencesList = new List<String>();
        frsSentencesList.Add(eastfrisian);
        gerSentencesList.Add(german);
    }

    String[] frsSentences = frsSentencesList.ToArray();
    String[] gerSentences = gerSentencesList.ToArray();

    Array.Resize(ref frsSentences, size);
    Array.Resize(ref gerSentences, size);

    return new String[][] { frsSentences, gerSentences };
}

String[][] SplitMeanings(string eastfrisian, string german)
{
    List<String> frsSentences = new List<String>();
    String[] gerSentences = german.Split(", ");

    for (int i = 0; i < gerSentences.Length; i++)
    {
        frsSentences.Add(eastfrisian);
    }

    return new String[][] { frsSentences.ToArray(), gerSentences };
}

string FirstCharToUpper(string input)
{
    if (String.IsNullOrEmpty(input))
    {
        Console.WriteLine(input);
        throw new ArgumentException("ARGH!");
    }
    return input.First().ToString().ToUpper() + input.Substring(1);
}

string AddPoint(string input)
{
    if (input.Last().ToString().Equals(".") || input.Last().ToString().Equals("?") || input.Last().ToString().Equals("!"))
    {
        return input;
    }

    return (input + ".");
}

string CutSentenceIfNecessary(string frs, string ger)
{
    if (ger.Contains(","))
    {
        int maxLength = (int)((float)(frs.Length) * 1.3f);
        if (ger.Length > maxLength)
        {
            return ger.Substring(0, ger.LastIndexOf(",")).Trim();
        }
    }
    return ger.Trim();
}

string CutBrackets(string gerString)
{
    string[] strs = gerString.Split("(");
    if (strs.Length > 1 && (strs[1].Contains("Rdw.: \"") || strs[1].Contains("Spr.: \"")))
    {
        gerString = strs[1].Split("\"")[1];
    }
    else
    {
        gerString = strs[0];
    }
    return gerString;
}

List<DictionaryRow> getDictionaryContent(String commandText)
{
    var dataBaseConnection = DataBaseConnection.GetConnection(MapPath(""));
    var sqlcmd = new SqliteCommand
    {
        Connection = dataBaseConnection
    };

    sqlcmd.CommandText = commandText;

    sqlcmd.Prepare();

    var reader = sqlcmd.ExecuteReader();
    var dictionaryRows = new List<DictionaryRow>();

    Console.Write("start reading dictionary");

    while (reader.Read())
    {
        var values = new DictionaryRow();
        values.ID = reader.GetInt64("ID");
        values.Ostfriesisch = reader.GetValue("Ostfriesisch").ToString();
        values.Deutsch = reader.GetValue("Deutsch").ToString();
        values.Englisch = reader.GetValue("Englisch").ToString();
        values.Artikel = reader.GetValue("Artikel").ToString();
        values.Nebenformen = reader.GetValue("Nebenformen").ToString();
        values.Standardform = reader.GetValue("Standardform").ToString();
        dictionaryRows.Add(values);
    }

    reader.Close();
    dataBaseConnection.Close();

    Console.Write("finished reading dictionary");
    return dictionaryRows;
}

static void Shuffle(List<List<String>> list)
{
    Random rng = new Random(SQL_SEED);
    int n = list.Count;
    while (n > 1)
    {
        n--;
        int k = rng.Next(n + 1);
        List<String> value = list[k];
        list[k] = list[n];
        list[n] = value;
    }
}

Console.WriteLine("Path: " + MapPath(""));

using (StreamWriter outputFileEastFrisian = new StreamWriter(MapPath("data/eastfrisian.txt")))
using (StreamWriter outputFileGerman = new StreamWriter(MapPath("data/german.txt")))
{
    var dictionaryRows = getDictionaryContent("SELECT * FROM WBFTS " +
        "WHERE Deutsch != '-' " +
        "AND Wortart = 'Phrase' " +
        "ORDER BY LENGTH(Ostfriesisch) DESC");
    for (int i = 0; i < 1; i++)
    {
        foreach (var dictionaryRow in dictionaryRows)
        {
            var frsString = dictionaryRow.Ostfriesisch;
            var gerString = dictionaryRow.Deutsch;

            gerString = gerString.Replace("\n", " ");
            frsString = frsString.Replace("\n", " ");

            gerString = gerString.Split(";")[0];
            gerString = CutBrackets(gerString);

            gerString = FilterAllAbbreviations(gerString);

            if (!IsToBeFiltered(frsString))
            {
                gerString = CutSentenceIfNecessary(frsString, gerString);
                String[][] sentences = SplitSentences(frsString, gerString);

                for (int j = 0; j < sentences[0].Length; j++)
                {
                    frsString = sentences[0][j];
                    gerString = sentences[1][j];
                    Console.WriteLine(frsString + " :  " + gerString);
                    outputFileGerman.WriteLine(gerString);
                    outputFileEastFrisian.WriteLine(frsString);
                }
            }
        }
    }

    dictionaryRows = getDictionaryContent(
        "SELECT *, " +
        "((ID * " + SQL_SEED + ") % 100000) AS RandomOrder " +
        "FROM WBFTS " +
        "WHERE Deutsch != '-' " +
        "AND Wortart = 'Phrase' " +
        "ORDER BY RandomOrder");
    for (int i = 0; i < 1; i++)
    {
        String finalGerString = "";
        String finalFrsString = "";
        bool first = false;
        foreach (var dictionaryRow in dictionaryRows)
        {
            var frsString = dictionaryRow.Ostfriesisch;
            var gerString = dictionaryRow.Deutsch;

            gerString = gerString.Replace("\n", " ");
            frsString = frsString.Replace("\n", " ");

            gerString = gerString.Split(";")[0];
            gerString = CutBrackets(gerString);

            gerString = FilterAllAbbreviations(gerString);

            if (!IsToBeFiltered(frsString))
            {
                if (!String.IsNullOrEmpty(frsString) && !String.IsNullOrEmpty(gerString))
                {
                    gerString = CutSentenceIfNecessary(frsString, gerString);

                    frsString = AddPoint(FirstCharToUpper(frsString));
                    gerString = AddPoint(FirstCharToUpper(gerString));
                    if (!first)
                    {
                        finalFrsString += frsString;
                        finalGerString += gerString;
                        Console.WriteLine(finalFrsString + "  :  " + finalGerString);
                        outputFileGerman.WriteLine(finalGerString);
                        outputFileEastFrisian.WriteLine(finalFrsString);

                        finalFrsString = "";
                        finalGerString = "";
                    }
                    else
                    {
                        finalFrsString += (frsString + " ");
                        finalGerString += (gerString + " ");
                    }

                    first = !first;
                }
            }
        }
    }

    // substantives
    dictionaryRows = getDictionaryContent(
        "SELECT *, " +
        "((ID * " + SQL_SEED + ") % 100000) AS RandomOrder " +
        "FROM WBFTS " +
        "WHERE Deutsch != '-' " +
        "AND Wortart = 'Substantiv' " +
        "ORDER BY RandomOrder " +
        "LIMIT 5000");

    foreach (var dictionaryRow in dictionaryRows)
    {
        var frsString = dictionaryRow.Ostfriesisch;
        var article = dictionaryRow.Artikel;
        var gerString = dictionaryRow.Deutsch;

        gerString = gerString.Replace("\n", " ");
        frsString = frsString.Replace("\n", " ");

        gerString = gerString.Split(";")[0];
        gerString = gerString.Split("(")[0];

        gerString = FilterAllAbbreviations(gerString);

        String[][] sentences = SplitMeanings(frsString, gerString);

        for (int j = 0; j < sentences[0].Length; j++)
        {
            frsString = sentences[0][j];
            gerString = sentences[1][j];

            if (gerString.Length < 25)
            {
                String[][] templateSentences = InsertIntoTemplate(frsString, gerString, new string[][] { new string[] { "[WORD] is mooj.", "[WORD] ist schön." },
                                                                                                         new string[] { "[WORD] is läif.", "[WORD] ist lieb." },
                                                                                                         new string[] { "[WORD] is interessant.", "[WORD] ist interessant." },
                                                                                                         new string[] { "[WORD] kan ik näit säin.", "[WORD] kann ich nicht sehen." } });
                for (int k = 0; k < templateSentences[0].Length; k++)
                {
                    frsString = templateSentences[0][k];
                    gerString = templateSentences[1][k];
                    Console.WriteLine(frsString + " :  " + gerString);
                    outputFileGerman.WriteLine(gerString);
                    outputFileEastFrisian.WriteLine(frsString);
                }
            }
        }
    }

    // verbs
    dictionaryRows = getDictionaryContent(
        "SELECT *, " +
        "((ID * " + SQL_SEED + ") % 100000) AS RandomOrder " +
        "FROM WBFTS " +
        "WHERE Deutsch != '-' " +
        "AND Wortart = 'Verb' " +
        "ORDER BY RandomOrder " +
        "LIMIT 5000");

    foreach (var dictionaryRow in dictionaryRows)
    {
        var frsString = dictionaryRow.Ostfriesisch;
        var gerString = dictionaryRow.Deutsch;

        gerString = gerString.Replace("\n", " ");
        frsString = frsString.Replace("\n", " ");

        gerString = gerString.Split(";")[0];
        gerString = gerString.Split("(")[0];

        gerString = FilterAllAbbreviations(gerString);

        String[][] sentences = SplitMeanings(frsString, gerString);

        for (int j = 0; j < sentences[0].Length; j++)
        {
            frsString = sentences[0][j];
            gerString = sentences[1][j];
            Console.WriteLine(frsString + " :  " + gerString);
            outputFileGerman.WriteLine(gerString);
            outputFileEastFrisian.WriteLine(frsString);
        }
    }

    // adjectives
    dictionaryRows = getDictionaryContent(
        "SELECT *, " +
        "((ID * " + SQL_SEED + ") % 100000) AS RandomOrder " +
        "FROM WBFTS " +
        "WHERE Deutsch != '-' " +
        "AND Wortart = 'Adjektiv' " +
        "ORDER BY RandomOrder " +
        "LIMIT 5000");

    foreach (var dictionaryRow in dictionaryRows)
    {
        var frsString = dictionaryRow.Ostfriesisch;
        var gerString = dictionaryRow.Deutsch;

        gerString = gerString.Replace("\n", " ");
        frsString = frsString.Replace("\n", " ");

        gerString = gerString.Split(";")[0];
        gerString = gerString.Split("(")[0];

        gerString = FilterAllAbbreviations(gerString);

        String[][] sentences = SplitMeanings(frsString, gerString);

        for (int j = 0; j < sentences[0].Length; j++)
        {
            frsString = sentences[0][j];
            gerString = sentences[1][j];
            Console.WriteLine(frsString + " :  " + gerString);
            outputFileGerman.WriteLine(gerString);
            outputFileEastFrisian.WriteLine(frsString);
        }
    }

    List<String> frsTexts = new List<String>();
    List<String> gerTexts = new List<String>();

    using (StreamReader inputFileTextEastFrisian = new StreamReader(MapPath("data/fan teksten/eastfrisian.txt")))
    using (StreamReader inputFileTextGerman = new StreamReader(MapPath("data/fan teksten/german.txt")))
    {
        String gerString = inputFileTextGerman.ReadLine();
        String frsString = inputFileTextEastFrisian.ReadLine();
        while (gerString != null)
        {
            frsTexts.Add(frsString);
            gerTexts.Add(gerString);
            for (int i = 0; i < MANUAL_DATASET_MUL; i++)
            {
                outputFileGerman.WriteLine(gerString);
                outputFileEastFrisian.WriteLine(frsString);
            }
            frsString = inputFileTextEastFrisian.ReadLine();
            gerString = inputFileTextGerman.ReadLine();
        }
    }

    List<List<String>> frsTextsList = new List<List<String>>();

    using (StreamWriter outputFileTextEastFrisian = new StreamWriter(MapPath("data/fan teksten/eastfrisian.txt")))
    using (StreamWriter outputFileTextGerman = new StreamWriter(MapPath("data/fan teksten/german.txt")))
    {
        for (int i = 0; i < frsTexts.Count; i++)
        {
            outputFileTextGerman.WriteLine(gerTexts[i]);
            outputFileTextEastFrisian.WriteLine(frsTexts[i]);

            frsTextsList.Add(new List<String>(new String[]{frsTexts[i], gerTexts[i]}));
        }

        Shuffle(frsTextsList);

        String finalGerString = "";
        String finalFrsString = "";
        bool first = false;
        foreach (List<String> text in frsTextsList)
        {
            var frsString = text[0];
            var gerString = text[1];

            gerString = gerString.Replace("\n", " ");
            frsString = frsString.Replace("\n", " ");

           if (!String.IsNullOrEmpty(frsString) && !String.IsNullOrEmpty(gerString))
            {
                gerString = CutSentenceIfNecessary(frsString, gerString);

                frsString = AddPoint(FirstCharToUpper(frsString));
                gerString = AddPoint(FirstCharToUpper(gerString));
                if (!first)
                {
                    finalFrsString += frsString;
                    finalGerString += gerString;
                    Console.WriteLine(finalFrsString + "  :  " + finalGerString);
                    outputFileGerman.WriteLine(finalGerString);
                    outputFileEastFrisian.WriteLine(finalFrsString);

                    finalFrsString = "";
                    finalGerString = "";
                }
                else
                {
                    finalFrsString += (frsString + " ");
                    finalGerString += (gerString + " ");
                }

                first = !first;
            }
        }



        /*while (true)
        {
            Console.WriteLine("");
            Console.WriteLine("Düütsk tekst:");
            String gerText = Console.ReadLine();
            if (gerText == "end")
            {
                break;
            }
            Console.WriteLine("Oostfräisk tekst:");
            String frsText = Console.ReadLine();
            Console.WriteLine("");

            String[][] sentences = SplitSentences(frsText, gerText);

            for (int j = 0; j < sentences[0].Length; j++)
            {
                String frsString = sentences[0][j];
                String gerString = sentences[1][j];
                Console.WriteLine(frsString + " :  " + gerString);
            }

            if (Console.ReadLine() != "stop")
            {
                Console.WriteLine("Saving...");
                for (int j = 0; j < sentences[0].Length; j++)
                {
                    String frsString = sentences[0][j];
                    String gerString = sentences[1][j];
                    outputFileGerman.WriteLine(gerString);
                    outputFileEastFrisian.WriteLine(frsString);
                    outputFileTextGerman.WriteLine(gerString);
                    outputFileTextEastFrisian.WriteLine(frsString);
                }
            }
        }*/
    }


}
