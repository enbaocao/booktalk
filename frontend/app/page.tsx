import Image from "next/image";

export default function Home() {
  return (
    <div className="pl-[10%] pr-[40%] pt-8 pb-8 text-2xl">
      <div className="space-y-6">
        <section id="introduction">
          <p className="text-gray-700 mb-4">
            An exploration of the distinct narrative voices within Helen Oyeyemi's novel, "White is for Witching," using computational stylometry.
          </p>
          <p className="mb-4">
            This project analyzes the text of the novel, identifying statistical patterns in writing style (like sentence length, vocabulary richness, pronoun usage, and punctuation) to cluster paragraphs likely belonging to different narrators or narrative modes.
          </p>
          <p className="mb-6">
            The goal is to gain insights into how Oyeyemi crafts the voices of Miranda, Eliot, the sentient house on Silver Road, and other narrative perspectives that contribute to the novel's unique and unsettling atmosphere.
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="summary">
          <h2 className="text-4xl font-bold mb-4 mt-8">Analysis Summary</h2>
          <p className="text-gray-600">(Summary content will be loaded here...)</p>
        </section>

        <hr className="border-gray-300"/>

        <section id="narrators">
          <h2 className="text-4xl font-bold mb-4 mt-8">Narrator Examples</h2>
          <p className="text-gray-600">(Narrator examples will be loaded here...)</p>
        </section>
      </div>

      <aside className="mt-8 space-y-6">
        <div className="p-4 border border-gray-300 rounded bg-white/50 shadow-sm">
          <h3 className="text-3xl font-semibold mb-4">Visualizations</h3>
          <p className="text-xl text-gray-600">(Charts and key stats will appear here...)</p>
        </div>
      </aside>
    </div>
  );
}
