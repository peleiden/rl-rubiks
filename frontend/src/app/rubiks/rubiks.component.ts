import { Component, OnInit } from '@angular/core';
import { CommonService } from '../common.service';

@Component({
  selector: 'app-rubiks',
  templateUrl: './rubiks.component.html',
  styleUrls: ["./rubiks.component.css"],
})
export class RubiksComponent implements OnInit {

  sides = ["F", "B", "T", "D", "L", "R"];
  indices = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  colours = ["red", "orange", "white", "yellow", "green", "blue"];

  constructor(public commonService: CommonService) { }

  ngOnInit(): void {
  }

  public getClass(i: number, j: number) {
    const colour = this.colours[this.commonService.state[i][j]];
    return `sticker ${colour}`;
  }

}
