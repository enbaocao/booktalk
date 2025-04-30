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
            For this project, I split the book up into paragraphs. For each paragraph, I computed the average sentence length, vocabulary richness, pronoun usage, punctuation frequency, and other statistics to cluster paragraphs likely belonging to different narrators. I ended up with 5 clusters, which I will discuss below. I will discuss what makes each cluster's narrative voice distinctive and possible reasons behind erroneous classifications.
          </p>
          <p className="mb-4 text-3xl">
            I removed paragraphs with dialogue inside, since the dialogue often comes from a different narrator compared to the rest of the paragraph. I also removed short paragraphs (less than 3 words), since these may be confusing. I was left with 744 paragraphs to cluster.
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="narrator-1">
          <h2 className="text-4xl font-bold mb-4 mt-8">Cluster 1: House and House-Adjacent</h2>
          
          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center font-bold">example quotes</p>
              <p className="text-gray-700 text-2xl">"Stupid, stupid; Lily had been warned not to go to Haiti." (House, 9)</p>
              <p className="text-gray-700 text-2xl">"But black wells only yield black water." (House, 10)</p>
              <p className="text-gray-700 text-2xl">"Courage, cabbage, cuttage, cottage." (House, 201)</p>
              <p className="text-gray-700 text-2xl">"Dad doesn't notice, but chairs are moved in the house. You leave a room and when you return the chair is scraped back from the table. Doors you leave closed are opened behind your back. And every day, the shoes." (Eliot, 282)</p>
              <p className="text-gray-700 text-2xl">"Step, step, halt." (ambiguous, 283)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            In this cluster, we mainly see the house's narration. One characteristic about the house's narration is the length of the sentences; they're heavy in commas and are generally shorter. The fourth quote isn't from the house, but Eliot uses a lot of house-like diction, with chairs and doors. The last quote is particularly interesting. It occurs in the final section of the book, when Eliot tries to solve Miranda's disappearance. The model identifies 'step, step, creak' as a noise from the house, rather than Miranda, which shows how Miranda has merged with the house.
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="narrator-2">
          <h2 className="text-4xl font-bold mb-4 mt-8">Cluster 2: Third Person Objective</h2>
          
          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center text-gray-700 text-3xl mb-4 font-bold">example quotes</p>
              <p className="text-gray-700 text-2xl">"Miranda Silver is in Dover, in the ground beneath her mother's house. Her throat is blocked with a slice of apple (to stop her speaking words that may betray her) her ears are filled with earth (to keep her from hearing sounds that will confuse her) her eyes are closed, but her heart thrums hard like hummingbird wings. She chose this as the only way to fight the soucouyant." (Ore, 1)</p>
              <p className="text-gray-700 text-2xl">"He wooed his wife with peach tarts he'd learnt from his pastry-maker father. The peaches fused into the dough with their skins intact, bittered and sweetened by burnt sugar." (third person, 14)</p>
              <p className="text-gray-700 text-2xl">"Miranda had been admitted to the clinic because one morning Eliot had found her wordless and thoughtful. It had been a long night, a perfect full moon tugging the sky around it into clumsy wrinkles. Miranda had been bleeding slightly from the scalp and her wrists were bound together with extreme dexterity and thin braids of her own hair." (third person, 29)</p>
              <p className="text-gray-700 text-2xl">"The letter read: Dear Miranda Silver, This house is bigger than you know! There are extra floors, with lots of people on them. They look at you, and they never move. We do not like them. We do not like this house, and we are glad to be going away." (Deme and Suryaz, 66)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            In this cluster, we see a mix of narrators, but all with an objective nature. Ore's declaration of Miranda's situation, the third person narrator's telling of Luc's baking and Miranda's past, and Deme and Suryaz's informational letter to Miranda are all based on facts rather than emotions.
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="narrator-3">
          <h2 className="text-4xl font-bold mb-4 mt-8">Cluster 3: Eliot</h2>
          
          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center text-gray-700 text-3xl mb-4 font-bold">example quotes</p>
              <p className="text-gray-700 text-2xl">"Miri and I were ten; Dad spent some time with a big map, planning a scenic route, and then he drove the moving van himself. Miri and I fidgeted at first, then settled when we saw cliffs bruising the skyline and smelt birds and wet salt on the air." (Eliot, 17)</p>
              <p className="text-gray-700 text-2xl">"We thought it would be hard to make friends because of the way people came out and stared at us in the moving van as it passed through the streets. But Miri is good at making friends, and I am good at tagging along on expeditions and acting as if the whole thing was my idea in the first place." (Eliot, 19)</p>
              <p className="text-gray-700 text-2xl">"Emma texted me: Jean de Bergieres—they searched for her in the oven(!) and found her in the attic." (Eliot, 19)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            In this cluster, we see Eliot's narration. Eliot narrates most of the book, and his narration is characterized by longer and more concrete sentences, compared to the other narrators. We see this with his description of seeing 'cliffs bruising the skyline' and smelling 'birds and wet salt.' His narration is tangible and direct, but not without imagery. He also stands out for his prominent mentions of Miranda as Miri, playing into the gothic motif of intertwined twins. He frequently refers to the pair, using 'Miri and I' and 'we' often in the quotes above. 
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="narrator-4">
          <h2 className="text-4xl font-bold mb-4 mt-8">Cluster 4: Personal Narration</h2>
          
          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center text-gray-700 text-3xl mb-4 font-bold">example quotes</p>
              <p className="text-gray-700 text-2xl">"Miri's accusations, her whole manner that night scared the shit out of me. She looked in my direction but she couldn't seem to focus on me. She was the thinnest I'd ever seen her. Her hands and head were the heaviest parts of her. She hugged herself, her fingers pinning her dress to her ribs. There was an odd smell to her, heavy and thick." (Eliot, 2)</p>
              <p className="text-gray-700 text-2xl">"The perfect person was a girl. Bobbed dark hair, black dress, pearls she was too young for, mouth, nose and chin familiar. The perfect person had beautifully shaped hands, but no fingernails. A swanlike neck that met the jaw at a devastating but impossible angle." (Eliot/Miranda, 56)</p>
              <p className="text-gray-700 text-2xl">"I cried and cried for an hour or so, unable to bear the sound of my voice, so shrill and pleading, but unable to stop the will of the wind wheeling through me, cold in my insides. That was the first and last time I've heard my own voice. I suppose I am frightening. But Anna Good couldn't hear me." (House, 26-27)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            We see a selection of different narrators, united by the personal nature of their narration. The first quote reflects Eliot's emotions, showing the subjective side of his narration. He expresses his fear of Miranda, focusing entirely on her. In the second quote, we again see narration focused on another person. This time, it's Miranda's idea of the perfect person. This is one of the rare moments we hear narration from Miranda through Eliot's understanding of her. In this quote, Miranda focuses completely on describing this perfect person, fixating on her appearance. Through this fixation, however, we also see Miranda's own feelings about herself. She feels inadequate, failing to live up to an impossible standard. In the final quote, the house reveals its suffering through narration as it witnesses Anna Good's struggle with pica. So while this cluster doesn't have a clear narrator, there is a common theme of subjective, personal narration.
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="narrator-5">
          <h2 className="text-4xl font-bold mb-4 mt-8">Cluster 5: Worry and Concern</h2>
          
          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center text-gray-700 text-3xl mb-4 font-bold">example quotes</p>
              <p className="text-gray-700 text-2xl">"Does she remember me at all I miss her I miss the way her eyes are the same shade of grey no matter the strength or weakness of the light I miss the taste of her I see her in my sleep, a star planted seed-deep, her arms outstretched, her fists clenched, her black dress clinging to her like mud." (Ore, 1)</p>
              <p className="text-gray-700 text-2xl">"We'd had an argument. Gusts of wind tangled in the apple trees around our house and dropped fruit onto the roof, made it sound like someone was tapping on the walls in the attic, Morse code for let me out, or something weirder. The argument was a stupid one that opened up a murky little mouth to take in other things. Principally it was about this pie I'd baked for her. She wouldn't eat any of it, and she wouldn't let me." (Eliot, 2)</p>
              <p className="text-gray-700 text-2xl">"That last time I saw Miri, she wasn't wearing any shoes. Five months ago I took that as security that she would come back." (Eliot, 2)</p>
              <p className="text-gray-700 text-2xl">"A part of me knows that we can't find her because something has happened to her." (Eliot, 3)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            Cluster 5, similar to Cluster 4, exhibits a mix of narrators. The common thread through them is a sense of worry or concern. We see Ore's yearning for Miranda post-disappearance and Eliot's anxiety-filled obsession with figuring out where Miranda is. 
          </p>
        </section>

        <hr className="border-gray-300"/>

        <section id="uncertainties">
          <h2 className="text-4xl font-bold mb-4 mt-8">Uncertainties</h2>
          
          <p className="mb-4 text-3xl">
            We'll go on to explore the quotes with the highest uncertainties. The model struggled assigning a cluster to these quotes below.
          </p>

          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center text-gray-700 text-3xl mb-4 font-bold">ambiguous quote #1</p>
              <p className="text-gray-700 text-2xl">"Miranda is at <em>home</em></p>
              <p className="text-gray-700 text-2xl">(homesick, home <em>sick</em>)</p>
              <p className="text-gray-700 text-2xl">Miranda can't come in today Miranda has a <em>condition</em> called pica she has eaten a great deal of chalk—she really can't help herself—she has been very ill—<em>Miranda has pica she can't come in</em> today, she is stretched out inside a wall she is feasting on plaster she has pica" (House, 3)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            This quote is narrated by the house, evident from the '29 barton road:' sentence right before it. However, without seeing that sentence, it makes sense that the model struggled to classify this quote. While the house takes on a malevolent, sadistic personality later in the book, it is surprisingly protective in this quote. The house explains Miranda's condition and sympathizes: "she really can't help herself." Reading this paragraph by itself, it almost seems narrated by Eliot or Ore, perhaps explaining Miranda's absence to Cambridge over the phone.
          </p>
          
          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center text-gray-700 text-3xl mb-4 font-bold">ambiguous quote #2</p>
              <p className="text-gray-700 text-2xl">"I find Luc interesting. He really has no idea what to do now, and because he is not mine I don't care about him. I do, however, take great delight in the power of a push, a false burst of light at the bottom of a cliff, just one little encouragement to the end." (House, 15-16)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            The beginning of this quote resembles Eliot's narration. The first two sentences are straightforward and concrete, characteristic of Eliot. However, there is a sudden change of tone, which likely led to the model's high uncertainty score. In the second half of the quote, we see the house's sadistic treatment of Luc, shown through its delight in giving Luc a 'false burst of light' and a 'little encouragement to the end.' This combination of two distinct narrative voices allowed this quote to avoid easy classification.
          </p>

          <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200">
            <div style={{ padding: '16px 40px' }} className="space-y-8">
              <p className="text-center text-gray-700 text-3xl mb-4 font-bold">ambiguous quote #3 and #4</p>
              <p className="text-gray-700 text-2xl">"But I wrote, Miri I'm lonely. I dropped the words onto the paper so hard that they're doubled by the thin perforations around them." (Eliot, 4)</p>              <p className="text-gray-700 text-2xl">"What I mean is, each act of speech stands on the belief that someone will hear. My note to Miri says more than just I'm lonely. Invisibly it says that I know she will see this, and that when she sees this it will turn her, turn her back, return her." (Eliot, 4)</p>
            </div>
          </div>
          <p className="mb-4 text-3xl">
            I think what introduces ambiguity here is Eliot's uncharacteristically figurative narration. There's the evocative phrase 'dropped the words onto the paper,' as if they were free flowing, and the rhythmic repetition in 'turn her, turn her back, return her' paints Eliot as poetic. It's worth noting that at this moment in the book, Eliot is faced with Miranda's sudden disappearance. He is left grasping for any sign of meaning. Perhaps this shift from matter-of-fact narration to a more figurative style shows Eliot's uncertainty with the world; he doesn't understand it, so he can no longer tell the story in a concrete way.
          </p>
          
        </section>

        <hr className="border-gray-300"/>

        <section id="conclusion">
          <h2 className="text-4xl font-bold mb-4 mt-8">Conclusion</h2>
          <p className="text-gray-600 text-3xl">
            Overall, the five clusters represented distinct narrative voices, albeit not necessarily different narrators. There were clear clusters for the omniscient third person narrator, the house, and Eliot, but the other clusters were more so based on style rather than narrator.
          </p>
          <p className="text-gray-600 text-3xl">
            In my opinion, the most interesting quotes were the ambiguous ones. Through these uncertain quotes, I saw the house's nature change depending on the section, accentuating its manipulative nature, as well as Eliot's growing uncertainty after Miranda's disappearance. It was fascinating to examine why the model struggled, and it revealed profound insights into the book.
          </p>
        </section>

        {/* Decorative insignia to mark end of content */}
        <div className="flex justify-center my-16">
          <div className="text-gray-400 text-5xl">
            <span className="px-4">❦</span>
          </div>
        </div>
        
        {/* Using a CSS grid approach for flexible vertical spacing */}
        <div className="grid grid-cols-1 gap-8">
          <div className="h-32"></div>
          <div className="h-32"></div>
        </div>

      </div>

    </div>
  );
}
