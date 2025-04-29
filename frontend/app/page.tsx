import Image from "next/image";

export default function Home() {
  return (
    <div className="pl-[10%] pr-[40%] pt-8 pb-8">
      <div className="space-y-6">
        <section id="introduction">
          <p className="text-gray-700 mb-4 text-3xl">
            I think the most fascinating aspect of Oyeyemi's White is For Witching isn't the multiple narrators, but specifically how characters fluidly hand off the narrative to each other. I found myself asking "who really is speaking?" often, and the answer enhanced my understanding of the book rather than taking away from it. Typographical cues, like a connecting phrase or capitalized questions that frame the narrative like "WHO DO YOU BELIEVE?" allow characters to take turns narrating. But most often, there is no transition; Oyeyemi throws you right in, letting you figure it out.
          </p>
          <p className="text-gray-700 mb-4 text-3xl">
            How do we figure out who is narrating? As humans, we have intuition. Sometimes, we can just tell. But unconsciously, we pick up on cues like tone, word choice, and sentence structure to distinguish between narrators. I wanted to see computers try to do the same.
          </p>
          <p className="mb-4 text-3xl">
            For this project, I split the book up into paragraphs. For each paragraph, I computed the average sentence length, vocabulary richness, pronoun usage, punctuation frequency, and other statistics to cluster paragraphs likely belonging to different narrators. I ended up with 6 main clusters, which I will discuss below. For each cluster, I will discuss what makes that narrative voice distinctive and possible reasons behind erroneous classifications.
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="summary">
          <h2 className="text-4xl font-bold mb-4 mt-8">Analysis Summary</h2>
          <p className="text-gray-600 text-3xl">(Summary content will be loaded here...)</p>
        </section>

        <hr className="border-gray-300"/>

        <section id="narrators">
          <h2 className="text-4xl font-bold mb-4 mt-8">Narrator Examples</h2>
          <p className="text-gray-600 text-3xl">(Narrator examples will be loaded here...)</p>
        </section>
      </div>

      <aside className="mt-8 space-y-6">
        <div className="p-4 border border-gray-300 rounded bg-white/50 shadow-sm">
          <h3 className="text-3xl font-semibold mb-4">Visualizations</h3>
          <p className="text-gray-600 text-3xl">(Charts and key stats will appear here...)</p>
        </div>
      </aside>
    </div>
  );
}
