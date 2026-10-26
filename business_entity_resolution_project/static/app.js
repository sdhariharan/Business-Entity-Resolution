const $=id=>document.getElementById(id);
["1","2","3"].forEach(n=>$( "f"+n).addEventListener("change",()=>$( "n"+n).textContent=$( "f"+n).files[0]?.name||"No file selected"));
$("run").addEventListener("click",async()=>{
 const err=$("error");err.classList.add("hidden");$("results").classList.add("hidden");
 if(!["1","2","3"].every(n=>$("f"+n).files.length)){err.textContent="Please upload all three TSV files.";err.classList.remove("hidden");return}
 const fd=new FormData();["1","2","3"].forEach(n=>fd.append("source"+n,$("f"+n).files[0]));
 $("run").disabled=true;$("loading").classList.remove("hidden");
 try{
  const r=await fetch("/api/match",{method:"POST",body:fd}),d=await r.json();
  if(!r.ok||!d.success)throw new Error(d.error||"Prediction failed.");
  $("s1").textContent=d.source1_records;$("s2").textContent=d.source2_records;$("s3").textContent=d.source3_records;$("m").textContent=d.total_matches;$("threshold").textContent=d.threshold;
  $("body").innerHTML=d.preview.map(x=>`<tr><td>${escapeHtml(x.source1_entity_id)}</td><td>${escapeHtml(x.matched_entity_ids||"No match")}</td></tr>`).join("");
  $("results").classList.remove("hidden");
 }catch(e){err.textContent=e.message;err.classList.remove("hidden")}finally{$("run").disabled=false;$("loading").classList.add("hidden")}
});
function escapeHtml(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}
